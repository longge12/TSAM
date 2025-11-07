"""
分支头模块
实现语义分支头（欧式拉近）和几何分支头（双曲拉远）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hyperbolic import exp0_lorentz, log0_lorentz, softplus_pos, lorentz_angle, lorentz_radius


class SemanticHead(nn.Module):
    """
    语义对齐分支头（欧式空间，拉近）
    支持 Cross-Attention 和双塔投影两种模式
    """
    def __init__(
        self,
        d: int,
        use_cross_attn: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.use_cross_attn = use_cross_attn
        
        if use_cross_attn:
            # Cross-Attention 模式
            self.cross_attn_t = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attn_v = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm_t = nn.LayerNorm(d)
            self.norm_v = nn.LayerNorm(d)
            self.pool = nn.AdaptiveAvgPool1d(1)  # 用于池化
        else:
            # 双塔投影模式
            self.proj_t = nn.Sequential(
                nn.Linear(d, d),
                nn.LayerNorm(d),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d, d)
            )
            self.proj_v = nn.Sequential(
                nn.Linear(d, d),
                nn.LayerNorm(d),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d, d)
            )
    
    def forward(self, e_t: torch.Tensor, e_v: torch.Tensor) -> tuple:
        """
        Args:
            e_t: (batch_size, d) 文本编码
            e_v: (batch_size, d) 视觉编码
        
        Returns:
            t_sem: (batch_size, d) 语义对齐后的文本表示
            v_sem: (batch_size, d) 语义对齐后的视觉表示
        """
        if self.use_cross_attn:
            # Cross-Attention 模式
            # 扩展维度以支持 attention: (batch, seq_len=1, d)
            e_t_seq = e_t.unsqueeze(1)  # (batch, 1, d)
            e_v_seq = e_v.unsqueeze(1)  # (batch, 1, d)
            
            # 文本到视觉的交叉注意力
            h_t_prime, _ = self.cross_attn_t(
                query=e_t_seq,
                key=e_v_seq,
                value=e_v_seq
            )
            h_t_prime = self.norm_t(h_t_prime + e_t_seq)
            
            # 视觉到文本的交叉注意力
            h_v_prime, _ = self.cross_attn_v(
                query=e_v_seq,
                key=e_t_seq,
                value=e_t_seq
            )
            h_v_prime = self.norm_v(h_v_prime + e_v_seq)
            
            # 池化（去掉序列维度）
            t_sem = h_t_prime.squeeze(1)  # (batch, d)
            v_sem = h_v_prime.squeeze(1)  # (batch, d)
        else:
            # 双塔投影模式
            t_sem = self.proj_t(e_t)
            v_sem = self.proj_v(e_v)
        
        return t_sem, v_sem


class HyperbolicHead(nn.Module):
    """
    几何对齐分支头（双曲空间，拉远）
    使用 Lorentz 模型，通过角度对齐和半径分层保持层级差异
    """
    def __init__(
        self,
        d: int,
        curvature_init: float = 0.1,
        learnable_curvature: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.learnable_curvature = learnable_curvature
        
        # 投影到切空间
        self.proj_t = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d)
        )
        
        # 可学习的曲率参数
        if learnable_curvature:
            self.raw_c_t = nn.Parameter(torch.tensor(curvature_init))
            self.raw_c_v = nn.Parameter(torch.tensor(curvature_init))
        else:
            self.register_buffer('raw_c_t', torch.tensor(curvature_init))
            self.register_buffer('raw_c_v', torch.tensor(curvature_init))
        
        # 改进：对齐初始化，让两个投影层的最后一层权重初始时相似
        # 这样可以避免一开始就方向完全相反
        self._init_aligned()
    
    def _init_aligned(self):
        """对齐初始化：让proj_t和proj_v的最后一层权重初始时相似"""
        # 获取最后一层Linear的权重
        t_last_layer = self.proj_t[-1]
        v_last_layer = self.proj_v[-1]
        
        # 使用相同的初始化策略，但添加小的随机扰动
        # 这样初始时方向相近，但仍有区分度
        with torch.no_grad():
            # 先初始化v_last_layer
            nn.init.xavier_uniform_(v_last_layer.weight)
            nn.init.zeros_(v_last_layer.bias)
            
            # t_last_layer使用类似的初始化，但添加小的扰动
            nn.init.xavier_uniform_(t_last_layer.weight)
            # 添加小的同向扰动（而不是完全随机）
            t_last_layer.weight.data = 0.8 * v_last_layer.weight.data.clone() + 0.2 * t_last_layer.weight.data
            nn.init.zeros_(t_last_layer.bias)
    
    def curvatures(self) -> tuple:
        """获取曲率参数（确保为正数）"""
        c_t = softplus_pos(self.raw_c_t)
        c_v = softplus_pos(self.raw_c_v)
        return c_t, c_v
    
    def forward(self, e_t: torch.Tensor, e_v: torch.Tensor) -> tuple:
        """
        Args:
            e_t: (batch_size, d) 文本编码
            e_v: (batch_size, d) 视觉编码
        
        Returns:
            t_hyp: (batch_size, d+1) 双曲空间中的文本表示
            v_hyp: (batch_size, d+1) 双曲空间中的视觉表示
            c_t: 文本曲率
            c_v: 视觉曲率
        """
        # 投影到切空间
        u_t = torch.tanh(self.proj_t(e_t))  # 稳定化
        u_v = torch.tanh(self.proj_v(e_v))
        
        # 获取曲率
        c_t, c_v = self.curvatures()
        
        # 指数映射到 Lorentz 空间
        t_hyp = exp0_lorentz(u_t, c_t)
        v_hyp = exp0_lorentz(u_v, c_v)
        
        return t_hyp, v_hyp, c_t, c_v
    
    def log_map(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """将对数映射暴露为方法，供融合模块使用"""
        return log0_lorentz(x, c)
    
    def compute_angle(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """计算两个向量的角度"""
        return lorentz_angle(x, y, c)
    
    def compute_radius(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """计算向量的半径"""
        return lorentz_radius(x, c)

