"""
融合模块
实现门控融合，将语义分支和几何分支的输出融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    门控融合模块
    使用门控机制融合语义分支（欧式）和几何分支（双曲）的输出
    """
    def __init__(
        self,
        d: int,
        use_diversity_reg: bool = False,
        diversity_margin: float = 0.5,
        dropout: float = 0.1,
        norm_inputs: bool = True,
    ):
        super().__init__()
        self.d = d
        self.use_diversity_reg = use_diversity_reg
        self.diversity_margin = diversity_margin
        self.norm_inputs = norm_inputs
        
        # 文本门控网络
        self.gate_t = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )
        
        # 视觉门控网络
        self.gate_v = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

        # 分支尺度校准：学得的正尺度，保证两路量纲对齐
        self.scale_t_sem_raw = nn.Parameter(torch.tensor(1.0))
        self.scale_t_hyp_raw = nn.Parameter(torch.tensor(1.0))
        self.scale_v_sem_raw = nn.Parameter(torch.tensor(1.0))
        self.scale_v_hyp_raw = nn.Parameter(torch.tensor(1.0))

        # 轻微的门控偏置：鼓励初期更多使用文本分支
        self.gate_bias_t = nn.Parameter(torch.tensor(0.2))
        self.gate_bias_v = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _pos_scale(x: torch.Tensor) -> torch.Tensor:
        # 正尺度（softplus 保证 >0）
        return F.softplus(x) + 1e-6
    
    def forward(
        self,
        t_sem: torch.Tensor,
        v_sem: torch.Tensor,
        t_hyp_tangent: torch.Tensor,
        v_hyp_tangent: torch.Tensor,
    ) -> tuple:
        """
        门控融合
        
        Args:
            t_sem: (batch_size, d) 语义分支的文本表示
            v_sem: (batch_size, d) 语义分支的视觉表示
            t_hyp_tangent: (batch_size, d) 几何分支的文本表示（切空间）
            v_hyp_tangent: (batch_size, d) 几何分支的视觉表示（切空间）
        
        Returns:
            t_fuse: (batch_size, d) 融合后的文本表示
            v_fuse: (batch_size, d) 融合后的视觉表示
            gate_info: dict 包含门控权重等信息
        """
        # 归一化并校准尺度，使两路在同一量纲
        if self.norm_inputs:
            t_sem = F.normalize(t_sem, p=2, dim=-1)
            v_sem = F.normalize(v_sem, p=2, dim=-1)
            t_hyp_tangent = F.normalize(t_hyp_tangent, p=2, dim=-1)
            v_hyp_tangent = F.normalize(v_hyp_tangent, p=2, dim=-1)

        s_t_sem = self._pos_scale(self.scale_t_sem_raw)
        s_t_hyp = self._pos_scale(self.scale_t_hyp_raw)
        s_v_sem = self._pos_scale(self.scale_v_sem_raw)
        s_v_hyp = self._pos_scale(self.scale_v_hyp_raw)

        t_sem_s = t_sem * s_t_sem
        t_hyp_tangent_s = t_hyp_tangent * s_t_hyp
        v_sem_s = v_sem * s_v_sem
        v_hyp_tangent_s = v_hyp_tangent * s_v_hyp

        # 计算门控权重（加入可学习偏置）
        # 门控去偏：仅用于gate_t的输入，对几何分支停止梯度，避免门控过度依赖几何而抑制文本
        gate_input_t = torch.cat([t_sem_s, t_hyp_tangent_s.detach()], dim=-1)
        gate_t_raw = self.gate_t(gate_input_t)
        gate_t = torch.sigmoid(gate_t_raw + self.gate_bias_t)  # (batch_size, 1)

        gate_input_v = torch.cat([v_sem_s, v_hyp_tangent_s], dim=-1)
        gate_v_raw = self.gate_v(gate_input_v)
        gate_v = torch.sigmoid(gate_v_raw + self.gate_bias_v)  # (batch_size, 1)

        # 门控融合（在切空间/欧式空间中）
        t_fuse = gate_t * t_sem_s + (1 - gate_t) * t_hyp_tangent_s
        v_fuse = gate_v * v_sem_s + (1 - gate_v) * v_hyp_tangent_s

        gate_info = {
            'gate_t': gate_t.squeeze(-1),  # (batch_size,)
            'gate_v': gate_v.squeeze(-1),  # (batch_size,)
            'gate_t_raw': gate_t_raw.squeeze(-1),
            'gate_v_raw': gate_v_raw.squeeze(-1),
            'scale_t_sem': s_t_sem.detach(),
            'scale_t_hyp': s_t_hyp.detach(),
            'scale_v_sem': s_v_sem.detach(),
            'scale_v_hyp': s_v_hyp.detach(),
        }

        return t_fuse, v_fuse, gate_info
    
    def compute_diversity_loss(
        self,
        t_sem: torch.Tensor,
        v_sem: torch.Tensor,
        t_hyp_tangent: torch.Tensor,
        v_hyp_tangent: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算差异性正则损失，确保两个分支的输出保持差异
        
        Args:
            t_sem, v_sem: 语义分支输出
            t_hyp_tangent, v_hyp_tangent: 几何分支输出（切空间）
        
        Returns:
            diversity_loss: 标量损失
        """
        if not self.use_diversity_reg:
            return torch.tensor(0.0, device=t_sem.device)
        
        # 计算欧式距离
        diff_t = torch.norm(t_sem - t_hyp_tangent, p=2, dim=-1)
        diff_v = torch.norm(v_sem - v_hyp_tangent, p=2, dim=-1)
        
        # 鼓励差异大于 margin
        loss_t = torch.mean(torch.clamp(self.diversity_margin - diff_t, min=0.0))
        loss_v = torch.mean(torch.clamp(self.diversity_margin - diff_v, min=0.0))
        
        diversity_loss = loss_t + loss_v
        return diversity_loss
    
    def compute_gate_reg_loss(
        self,
        gate_info: dict,
        target_t: float = 0.5,
        target_v: float = 0.5,
        difficulty_aware: bool = False,
    ) -> torch.Tensor:
        """
        计算门控正则损失
        
        Args:
            gate_info: 门控信息字典
            difficulty_aware: 是否使用难度感知正则
        
        Returns:
            gate_reg_loss: 标量损失
        """
        gate_t = gate_info['gate_t']
        gate_v = gate_info['gate_v']
        
        if difficulty_aware:
            # 难度感知：鼓励难样本偏几何（门控值小），易样本偏语义（门控值大）
            # 这里使用简单的熵正则，鼓励门控值不要过于极端
            entropy_t = -gate_t * torch.log(gate_t + 1e-8) - (1 - gate_t) * torch.log(1 - gate_t + 1e-8)
            entropy_v = -gate_v * torch.log(gate_v + 1e-8) - (1 - gate_v) * torch.log(1 - gate_v + 1e-8)
            gate_reg_loss = -torch.mean(entropy_t + entropy_v)  # 负熵，鼓励多样性
        else:
            # 简单的 L2 正则，防止门控值过于极端
            gate_reg_loss = torch.mean((gate_t - target_t) ** 2) + torch.mean((gate_v - target_v) ** 2)
        
        return gate_reg_loss

