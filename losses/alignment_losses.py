"""
对齐损失模块
实现语义分支的 InfoNCE 损失和几何分支的角度损失、半径正则
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hyperbolic import lorentz_angle, lorentz_radius, lorentz_inner, lorentz_norm_sq


def info_nce_t2v_sem(
    t_sem: torch.Tensor,
    v_sem: torch.Tensor,
    temperature: float = 0.07,
    negative_samples: int = 32,  # 默认负样本数量，避免内存爆炸
) -> torch.Tensor:
    """
    语义分支的 InfoNCE 损失（文本-视觉对齐）
    
    Args:
        t_sem: (batch_size, d) 语义分支的文本表示
        v_sem: (batch_size, d) 语义分支的视觉表示
        temperature: 温度参数
        negative_samples: 负样本数量（默认32，避免内存爆炸；如果为None或>=batch_size-1，则使用全batch）
    
    Returns:
        loss: 标量损失
    """
    # L2 归一化
    t_sem = F.normalize(t_sem, p=2, dim=-1)
    v_sem = F.normalize(v_sem, p=2, dim=-1)
    
    batch_size = t_sem.size(0)
    
    # 如果负样本数量足够大或未指定，使用全batch（但batch_size较小时）
    use_full_batch = (negative_samples is None) or (negative_samples >= batch_size - 1) or (batch_size <= 64)
    
    if use_full_batch:
        # 原始方法：使用全batch（适用于小batch_size）
        sim_matrix = torch.matmul(t_sem, v_sem.t()) / temperature  # (batch_size, batch_size)
        labels = torch.arange(batch_size, device=t_sem.device)
        loss_t2v = F.cross_entropy(sim_matrix, labels)
        loss_v2t = F.cross_entropy(sim_matrix.t(), labels)
        loss = (loss_t2v + loss_v2t) / 2.0
    else:
        # 优化版：向量化批量负采样，避免Python循环（CPU瓶颈）
        actual_num_neg = min(negative_samples, batch_size - 1)
        
        # 计算正样本相似度（对角线）
        pos_sim = (t_sem * v_sem).sum(dim=-1) / temperature  # (batch_size,)
        
        # 优化：使用共享负样本池策略，完全避免Python循环（CPU瓶颈）
        # 策略：为整个batch生成一个共享的负样本池，所有样本都从这个池中采样
        # 这样可以完全向量化，无需Python循环
        all_indices = torch.arange(batch_size, device=t_sem.device)
        
        # 生成共享负样本池：随机选择actual_num_neg个样本作为所有样本的负样本池
        # 这样每个样本都使用相同的负样本集合（简化版，但完全向量化）
        if batch_size > actual_num_neg:
            # 随机选择actual_num_neg个样本作为共享负样本池
            perm_pool = torch.randperm(batch_size, device=t_sem.device)
            shared_neg_pool = perm_pool[:actual_num_neg]  # (actual_num_neg,)
        else:
            # 如果batch_size太小，使用所有其他样本
            shared_neg_pool = all_indices[:actual_num_neg]
        
        # 为每个样本构建负样本索引（排除自己）
        # 如果共享池中包含自己，替换为其他样本
        neg_indices_matrix = shared_neg_pool.unsqueeze(0).expand(batch_size, -1)  # (batch_size, actual_num_neg)
        
        # 处理对角线：向量化替换（避免Python循环）
        # 找到所有需要替换的位置（负样本池中包含自己的样本）
        self_mask = neg_indices_matrix == all_indices.unsqueeze(1)  # (batch_size, actual_num_neg)
        needs_replace = self_mask.any(dim=1)  # (batch_size,) - 哪些样本需要替换
        
        if needs_replace.any():
            # 找到不在负样本池中的候选样本
            candidates = all_indices[~torch.isin(all_indices, shared_neg_pool)]
            if len(candidates) > 0:
                # 为每个需要替换的样本随机选择一个候选
                num_replace = needs_replace.sum().item()
                replacements = candidates[torch.randint(len(candidates), (num_replace,), device=t_sem.device)]
                
                # 向量化替换：找到每个样本的第一个需要替换的位置
                replace_indices = self_mask[needs_replace].int().argmax(dim=1)  # 每个样本的第一个匹配位置
                neg_indices_matrix[needs_replace, replace_indices] = replacements
        
        # 向量化批量计算（全部在GPU上，避免CPU循环）
        chunk_size = 1024  # 增大chunk size以减少循环次数
        loss_t2v_list = []
        loss_v2t_list = []
        
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk_size_actual = chunk_end - chunk_start
            
            chunk_t = t_sem[chunk_start:chunk_end]  # (chunk_size, d)
            chunk_v = v_sem[chunk_start:chunk_end]  # (chunk_size, d)
            chunk_pos_sim = pos_sim[chunk_start:chunk_end]  # (chunk_size,)
            chunk_neg_indices = neg_indices_matrix[chunk_start:chunk_end]  # (chunk_size, actual_num_neg)
            
            # 向量化批量提取负样本（GPU操作，不是CPU循环）
            chunk_neg_v = v_sem[chunk_neg_indices]  # (chunk_size, actual_num_neg, d)
            chunk_neg_t = t_sem[chunk_neg_indices]  # (chunk_size, actual_num_neg, d)
            
            # 向量化批量计算相似度（全部在GPU上）
            neg_sim_t2v_chunk = (chunk_t.unsqueeze(1) * chunk_neg_v).sum(dim=-1) / temperature  # (chunk_size, actual_num_neg)
            neg_sim_v2t_chunk = (chunk_v.unsqueeze(1) * chunk_neg_t).sum(dim=-1) / temperature  # (chunk_size, actual_num_neg)
            
            # 批量构建logits
            logits_t2v_chunk = torch.cat([chunk_pos_sim.unsqueeze(1), neg_sim_t2v_chunk], dim=1)  # (chunk_size, 1 + actual_num_neg)
            logits_v2t_chunk = torch.cat([chunk_pos_sim.unsqueeze(1), neg_sim_v2t_chunk], dim=1)  # (chunk_size, 1 + actual_num_neg)
            
            # 批量计算损失（向量化，不是循环）
            labels_chunk = torch.zeros(chunk_size_actual, dtype=torch.long, device=t_sem.device)
            loss_t2v_chunk = F.cross_entropy(logits_t2v_chunk, labels_chunk)
            loss_v2t_chunk = F.cross_entropy(logits_v2t_chunk, labels_chunk)
            
            loss_t2v_list.append(loss_t2v_chunk)
            loss_v2t_list.append(loss_v2t_chunk)
        
        # 平均所有chunk的损失
        loss_t2v = torch.stack(loss_t2v_list).mean()
        loss_v2t = torch.stack(loss_v2t_list).mean()
        loss = (loss_t2v + loss_v2t) / 2.0
    
    return loss


def angle_loss_hyp(
    t_hyp: torch.Tensor,
    v_hyp: torch.Tensor,
    c: torch.Tensor,
    margin: float = 0.1,
    num_neg_samples: int = 32,  # 新增：负样本采样数量
) -> torch.Tensor:
    """
    几何分支的角度损失
    最小化正样本夹角，最大化负样本夹角
    
    Args:
        t_hyp: (batch_size, d+1) 双曲空间中的文本表示
        v_hyp: (batch_size, d+1) 双曲空间中的视觉表示
        c: 曲率参数（标量或 (batch_size,)）
        margin: 负样本角度 margin
        num_neg_samples: 每个正样本采样的负样本数量（默认32，避免内存爆炸）
    
    Returns:
        loss: 标量损失
    """
    batch_size = t_hyp.size(0)
    
    # 改进：直接使用cos_angle计算损失，避免acos在边界处的梯度消失问题
    # 计算Lorentz内积和范数
    inner = lorentz_inner(t_hyp, v_hyp)
    t_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(t_hyp), min=1e-10))
    v_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(v_hyp), min=1e-10))
    # 关键修复：Lorentz空间角度公式需要负号！
    # cos(α) = -⟨x, y⟩_H / (||x||_H * ||y||_H)
    cos_angle = torch.clamp(-inner / (t_norm * v_norm), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    
    # 正样本损失：最大化cos_angle（等价于最小化角度，但梯度更稳定）
    # 使用平滑的损失函数，避免在边界处梯度消失
    # 当 cos_angle = -1 时，损失 = 2.0；当 cos_angle = 1 时，损失 = 0.0
    # 使用 (1 - cos_angle)² 可以获得更强的梯度信号
    pos_loss = torch.mean((1.0 - cos_angle) ** 2)
    
    # 负样本角度（采样负样本，避免内存爆炸）- 向量化优化版
    neg_loss = torch.tensor(0.0, device=t_hyp.device)
    if batch_size > 1:
        # 限制负样本数量，避免在batch_size很大时内存爆炸
        actual_num_neg = min(num_neg_samples, batch_size - 1)
        
        # 优化：使用共享负样本池策略（与InfoNCE相同），避免Python循环
        all_indices = torch.arange(batch_size, device=t_hyp.device)
        
        # 生成共享负样本池
        if batch_size > actual_num_neg:
            perm_pool = torch.randperm(batch_size, device=t_hyp.device)
            shared_neg_pool = perm_pool[:actual_num_neg]
        else:
            shared_neg_pool = all_indices[:actual_num_neg]
        
        # 为每个样本构建负样本索引
        neg_indices_matrix = shared_neg_pool.unsqueeze(0).expand(batch_size, -1)
        
        # 向量化处理对角线（避免Python循环）
        self_mask = neg_indices_matrix == all_indices.unsqueeze(1)
        needs_replace = self_mask.any(dim=1)
        
        if needs_replace.any():
            candidates = all_indices[~torch.isin(all_indices, shared_neg_pool)]
            if len(candidates) > 0:
                num_replace = needs_replace.sum().item()
                replacements = candidates[torch.randint(len(candidates), (num_replace,), device=t_hyp.device)]
                replace_indices = self_mask[needs_replace].int().argmax(dim=1)
                neg_indices_matrix[needs_replace, replace_indices] = replacements
        
        # 向量化批量计算负样本角度（全部在GPU上）
        t_hyp_expanded = t_hyp.unsqueeze(1).expand(-1, actual_num_neg, -1)  # (batch_size, actual_num_neg, d+1)
        v_neg = v_hyp[neg_indices_matrix]  # (batch_size, actual_num_neg, d+1)
        
        # 处理曲率参数
        c_expanded = c if isinstance(c, torch.Tensor) and c.dim() > 0 else c
        if not isinstance(c_expanded, torch.Tensor):
            c_expanded = torch.tensor(c_expanded, device=t_hyp.device)
        if c_expanded.dim() == 0:
            c_expanded = c_expanded.expand(batch_size, actual_num_neg)
        elif c_expanded.dim() == 1:
            c_expanded = c_expanded.unsqueeze(1).expand(-1, actual_num_neg)
        
        # 批量计算负样本的cos_angle（向量化，在GPU上）
        t_hyp_flat = t_hyp_expanded.reshape(-1, t_hyp.size(-1))
        v_neg_flat = v_neg.reshape(-1, v_hyp.size(-1))
        c_expanded_flat = c_expanded.reshape(-1)
        
        # 计算负样本的Lorentz内积和范数
        neg_inner = lorentz_inner(t_hyp_flat, v_neg_flat)
        t_norm_neg = torch.sqrt(torch.clamp(-lorentz_norm_sq(t_hyp_flat), min=1e-10))
        v_norm_neg = torch.sqrt(torch.clamp(-lorentz_norm_sq(v_neg_flat), min=1e-10))
        # 关键修复：Lorentz空间角度公式需要负号！
        neg_cos_angle = torch.clamp(-neg_inner / (t_norm_neg * v_norm_neg), min=-1.0 + 1e-6, max=1.0 - 1e-6)
        neg_cos_angle = neg_cos_angle.reshape(batch_size, actual_num_neg)
        
        # 负样本损失：鼓励cos_angle接近 -1（角度接近π）
        # 目标：neg_cos_angle → -1，使用 (neg_cos_angle + 1)² 作为损失
        # 这样即使角度接近π，也有有效的梯度信号
        target_cos = torch.tensor(-1.0 + margin / 3.14159, device=t_hyp.device)  # 对应角度 ≈ π - margin
        neg_loss = torch.mean((neg_cos_angle - target_cos) ** 2)
    
    loss = pos_loss + neg_loss
    return loss


def radius_reg_hyp(
    t_hyp: torch.Tensor,
    v_hyp: torch.Tensor,
    c_t: torch.Tensor,
    c_v: torch.Tensor,
    r_t_target: float = 0.5,
    r_v_target: float = 1.0,
) -> torch.Tensor:
    """
    几何分支的半径正则
    鼓励文本更抽象（半径小），视觉更具体（半径大）
    
    Args:
        t_hyp: (batch_size, d+1) 双曲空间中的文本表示
        v_hyp: (batch_size, d+1) 双曲空间中的视觉表示
        c_t: 文本曲率
        c_v: 视觉曲率
        r_t_target: 目标文本半径
        r_v_target: 目标视觉半径
    
    Returns:
        loss: 标量损失
    """
    # 计算平均半径
    r_t = lorentz_radius(t_hyp, c_t)  # (batch_size,)
    r_v = lorentz_radius(v_hyp, c_v)  # (batch_size,)
    
    r_t_mean = torch.mean(r_t)
    r_v_mean = torch.mean(r_v)
    
    # L2 正则：鼓励接近目标半径
    loss_t = (r_t_mean - r_t_target) ** 2
    loss_v = (r_v_mean - r_v_target) ** 2
    
    # 确保 r_t < r_v
    constraint_loss = torch.clamp(r_t_mean - r_v_mean, min=0.0) ** 2
    
    loss = loss_t + loss_v + constraint_loss
    return loss


def compute_front_loss(
    t_sem: torch.Tensor,
    v_sem: torch.Tensor,
    t_hyp: torch.Tensor,
    v_hyp: torch.Tensor,
    c_t: torch.Tensor,
    c_v: torch.Tensor,
    lambda_h: float = 1.0,
    lambda_r: float = 0.1,
    tau: float = 0.07,
    margin: float = 0.1,
    r_t_target: float = 0.5,
    r_v_target: float = 1.0,
    num_neg_samples: int = 32,  # 新增：负样本采样数量
    vis_available: torch.Tensor = None,  # 新增：视觉模态可用性mask (num_ent,)
    txt_available: torch.Tensor = None,  # 新增：文本模态可用性mask (num_ent,)
) -> dict:
    """
    计算前端总损失
    
    Args:
        t_sem, v_sem: 语义分支输出
        t_hyp, v_hyp: 几何分支输出（双曲空间）
        c_t, c_v: 曲率参数
        lambda_h: 几何损失权重
        lambda_r: 半径正则权重
        tau: InfoNCE 温度参数
        margin: 角度损失 margin
        r_t_target, r_v_target: 目标半径
        vis_available: 视觉模态可用性mask (True表示可用，False表示缺失)
        txt_available: 文本模态可用性mask (True表示可用，False表示缺失)
    
    Returns:
        loss_dict: 包含各项损失的字典
    """
    # 模态缺失处理：只对同时有文本和视觉的样本计算损失
    if vis_available is not None and txt_available is not None:
        # 同时有文本和视觉的样本mask
        both_available = vis_available & txt_available  # (num_ent,)
        
        # 如果没有任何样本同时有文本和视觉，返回零损失
        if not both_available.any():
            return {
                'L_front': torch.tensor(0.0, device=t_sem.device),
                'L_sem': torch.tensor(0.0, device=t_sem.device),
                'L_ang': torch.tensor(0.0, device=t_sem.device),
                'L_rad': torch.tensor(0.0, device=t_sem.device),
            }
        
        # 只对同时有文本和视觉的样本计算损失
        t_sem_valid = t_sem[both_available]
        v_sem_valid = v_sem[both_available]
        t_hyp_valid = t_hyp[both_available]
        v_hyp_valid = v_hyp[both_available]
        c_t_valid = c_t if isinstance(c_t, torch.Tensor) and c_t.dim() > 0 else c_t
        c_v_valid = c_v if isinstance(c_v, torch.Tensor) and c_v.dim() > 0 else c_v
        
        # 如果c_t/c_v是标量，保持不变；如果是向量，需要mask
        if isinstance(c_t_valid, torch.Tensor) and c_t_valid.dim() > 0:
            c_t_valid = c_t_valid[both_available]
        if isinstance(c_v_valid, torch.Tensor) and c_v_valid.dim() > 0:
            c_v_valid = c_v_valid[both_available]
        
        # 语义损失（InfoNCE）- 只对有效样本
        L_sem = info_nce_t2v_sem(t_sem_valid, v_sem_valid, temperature=tau, negative_samples=num_neg_samples)
        
        # 几何损失（角度 + 半径正则）- 只对有效样本
        c_mean = (c_t_valid + c_v_valid) / 2.0 if isinstance(c_t_valid, torch.Tensor) else (c_t_valid + c_v_valid) / 2.0
        L_ang = angle_loss_hyp(t_hyp_valid, v_hyp_valid, c_mean, margin=margin, num_neg_samples=num_neg_samples)
        L_rad = radius_reg_hyp(t_hyp_valid, v_hyp_valid, c_t_valid, c_v_valid, r_t_target=r_t_target, r_v_target=r_v_target)
    else:
        # 如果没有提供模态可用性mask，使用所有样本（向后兼容）
        L_sem = info_nce_t2v_sem(t_sem, v_sem, temperature=tau, negative_samples=num_neg_samples)
        c_mean = (c_t + c_v) / 2.0 if isinstance(c_t, torch.Tensor) else (c_t + c_v) / 2.0
        L_ang = angle_loss_hyp(t_hyp, v_hyp, c_mean, margin=margin, num_neg_samples=num_neg_samples)
        L_rad = radius_reg_hyp(t_hyp, v_hyp, c_t, c_v, r_t_target=r_t_target, r_v_target=r_v_target)
    
    # 前端总损失
    L_front = L_sem + lambda_h * (L_ang + lambda_r * L_rad)
    
    loss_dict = {
        'L_sem': L_sem,
        'L_ang': L_ang,
        'L_rad': L_rad,
        'L_front': L_front,
    }
    
    return loss_dict

