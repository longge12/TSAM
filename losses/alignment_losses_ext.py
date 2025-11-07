"""
扩展的对齐损失模块（稳定角度拉远）
- 提供 hinge 型角度损失 + 难负样本挖掘 + 目标退火接口
- 复用原有的 InfoNCE 与半径正则
"""
import math
import torch
import torch.nn.functional as F

from .alignment_losses import info_nce_t2v_sem, radius_reg_hyp
from utils.hyperbolic import lorentz_inner, lorentz_norm_sq, log0_lorentz
from typing import Optional


@torch.no_grad()
def _ensure_scalar(x, device, dtype):
    return torch.tensor(float(x), device=device, dtype=dtype)


def angle_loss_hyp(
    t_hyp: torch.Tensor,
    v_hyp: torch.Tensor,
    c: torch.Tensor,
    margin: float = 0.1,  # 兼容旧接口（mse模式）
    num_neg_samples: int = 32,
    loss_type: str = 'hinge',
    pos_margin_rad: float = None,
    neg_margin_rad: float = None,
    use_hard_neg: bool = True,
    hard_neg_k: int = 8,
    return_stats: bool = False,
):
    """
    稳定的角度损失：
      - 正样本：角度 ≤ pos_margin_rad（默认为10°）
      - 负样本：角度 ≥ π - neg_margin_rad（默认为100°）

    当 loss_type='hinge' 时，超过阈值后损失为0；
    当 loss_type='mse' 时，回归到阈值对应的cos目标。
    """
    device = t_hyp.device
    dtype = t_hyp.dtype
    batch_size = t_hyp.size(0)

    # 正样本 cos（Lorentz：必须加负号）
    inner = lorentz_inner(t_hyp, v_hyp)
    t_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(t_hyp), min=1e-10))
    v_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(v_hyp), min=1e-10))
    cos_pos = torch.clamp(-inner / (t_norm * v_norm), min=-1.0 + 1e-6, max=1.0 - 1e-6)

    # 正样本：默认10°
    if pos_margin_rad is None:
        pos_margin_rad = 10.0 * math.pi / 180.0
    cos_pos_thresh = math.cos(pos_margin_rad)
    if loss_type == 'hinge':
        pos_loss = torch.relu(_ensure_scalar(cos_pos_thresh, device, cos_pos.dtype) - cos_pos).mean()
    else:
        pos_loss = ((1.0 - cos_pos) ** 2).mean()

    # 负样本
    neg_loss = torch.tensor(0.0, device=device, dtype=dtype)
    neg_cos_mean = torch.tensor(0.0, device=device, dtype=dtype)
    if batch_size > 1:
        actual_num_neg = min(num_neg_samples, batch_size - 1)

        all_indices = torch.arange(batch_size, device=device)
        if batch_size > actual_num_neg:
            shared_neg_pool = torch.randperm(batch_size, device=device)[:actual_num_neg]
        else:
            shared_neg_pool = all_indices[:actual_num_neg]

        neg_indices_matrix = shared_neg_pool.unsqueeze(0).expand(batch_size, -1)
        self_mask = neg_indices_matrix == all_indices.unsqueeze(1)
        needs_replace = self_mask.any(dim=1)
        if needs_replace.any():
            candidates = all_indices[~torch.isin(all_indices, shared_neg_pool)]
            if len(candidates) > 0:
                num_replace = int(needs_replace.sum().item())
                replacements = candidates[torch.randint(len(candidates), (num_replace,), device=device)]
                replace_indices = self_mask[needs_replace].int().argmax(dim=1)
                neg_indices_matrix[needs_replace, replace_indices] = replacements

        # 批量计算 neg cos
        t_hyp_expanded = t_hyp.unsqueeze(1).expand(-1, actual_num_neg, -1)
        v_neg = v_hyp[neg_indices_matrix]

        t_hyp_flat = t_hyp_expanded.reshape(-1, t_hyp.size(-1))
        v_neg_flat = v_neg.reshape(-1, v_hyp.size(-1))

        neg_inner = lorentz_inner(t_hyp_flat, v_neg_flat)
        t_norm_neg = torch.sqrt(torch.clamp(-lorentz_norm_sq(t_hyp_flat), min=1e-10))
        v_norm_neg = torch.sqrt(torch.clamp(-lorentz_norm_sq(v_neg_flat), min=1e-10))
        neg_cos = torch.clamp(-neg_inner / (t_norm_neg * v_norm_neg), min=-1.0 + 1e-6, max=1.0 - 1e-6)
        neg_cos = neg_cos.reshape(batch_size, actual_num_neg)
        neg_cos_mean = neg_cos.mean()

        # 负样本阈值：默认100° => angle_neg ≥ π-100°(≈80°) => cos ≤ cos(π-100°) = -cos(100°)
        if neg_margin_rad is None:
            neg_margin_rad = 100.0 * math.pi / 180.0
        cos_neg_thresh = math.cos(math.pi - neg_margin_rad)

        if loss_type == 'hinge':
            if use_hard_neg and hard_neg_k is not None and hard_neg_k > 0:
                k = min(hard_neg_k, actual_num_neg)
                topk_vals, _ = torch.topk(neg_cos, k=k, dim=1, largest=True, sorted=False)
                neg_loss = torch.relu(topk_vals - _ensure_scalar(cos_neg_thresh, device, topk_vals.dtype)).mean()
            else:
                neg_loss = torch.relu(neg_cos - _ensure_scalar(cos_neg_thresh, device, neg_cos.dtype)).mean()
        else:
            target_cos = _ensure_scalar(cos_neg_thresh, device, neg_cos.dtype)
            neg_loss = ((neg_cos - target_cos) ** 2).mean()

    loss = pos_loss + neg_loss
    if return_stats:
        stats = {
            'pos_loss': pos_loss.detach(),
            'neg_loss': neg_loss.detach(),
            'pos_cos_mean': cos_pos.mean().detach(),
            'neg_cos_mean': neg_cos_mean.detach(),
        }
        return loss, stats
    return loss


def angle_loss_tangent(
    u_t: torch.Tensor,
    u_v: torch.Tensor,
    num_neg_samples: int = 32,
    loss_type: str = 'hinge',
    pos_margin_rad: float = None,
    neg_margin_rad: float = None,
    use_hard_neg: bool = True,
    hard_neg_k: int = 8,
    return_stats: bool = False,
    # 记忆库难负
    memory_bank: Optional[object] = None,
    neg_source: str = 'hybrid',  # 'batch' | 'memory' | 'hybrid'
    memory_cand_size: int = 4096,
    memory_ratio: float = 0.5,
):
    """
    在同一切空间的欧式角度损失（推荐）
    - 正样本：cos(u_t, u_v) ≥ cos(pos_margin)
    - 负样本：cos(u_t, u_v_neg) ≤ cos(pi - neg_margin)
    """
    device = u_t.device
    dtype = u_t.dtype
    batch_size = u_t.size(0)

    u_t_n = torch.nn.functional.normalize(u_t, p=2, dim=-1)
    u_v_n = torch.nn.functional.normalize(u_v, p=2, dim=-1)

    cos_pos = (u_t_n * u_v_n).sum(dim=-1)

    if pos_margin_rad is None:
        pos_margin_rad = 10.0 * math.pi / 180.0
    cos_pos_thresh = math.cos(pos_margin_rad)
    if loss_type == 'hinge':
        pos_loss = torch.relu(_ensure_scalar(cos_pos_thresh, device, cos_pos.dtype) - cos_pos).mean()
    else:
        pos_loss = ((1.0 - cos_pos) ** 2).mean()

    neg_loss = torch.tensor(0.0, device=device, dtype=dtype)
    neg_cos_mean = torch.tensor(0.0, device=device, dtype=dtype)
    if batch_size > 1:
        actual_num_neg = min(num_neg_samples, batch_size - 1)

        all_indices = torch.arange(batch_size, device=device)
        if batch_size > actual_num_neg:
            shared_neg_pool = torch.randperm(batch_size, device=device)[:actual_num_neg]
        else:
            shared_neg_pool = all_indices[:actual_num_neg]

        neg_indices_matrix = shared_neg_pool.unsqueeze(0).expand(batch_size, -1)
        self_mask = neg_indices_matrix == all_indices.unsqueeze(1)
        needs_replace = self_mask.any(dim=1)
        if needs_replace.any():
            candidates = all_indices[~torch.isin(all_indices, shared_neg_pool)]
            if len(candidates) > 0:
                num_replace = int(needs_replace.sum().item())
                replacements = candidates[torch.randint(len(candidates), (num_replace,), device=device)]
                replace_indices = self_mask[needs_replace].int().argmax(dim=1)
                neg_indices_matrix[needs_replace, replace_indices] = replacements

        # 批内候选
        u_v_neg_batch = u_v_n[neg_indices_matrix]  # (batch, K, d)
        u_t_neg = u_t_n.unsqueeze(1).expand(-1, actual_num_neg, -1)

        # 记忆库候选（跨批）
        cos_list = []
        src_list = []
        if neg_source in ('memory', 'hybrid') and memory_bank is not None:
            mem_k = hard_neg_k if neg_source == 'memory' else max(1, int(hard_neg_k * memory_ratio))
            mem_negs = memory_bank.sample_hard_neg(u_t_n, from_mod='v', topk=mem_k, cand_size=memory_cand_size)
            if mem_negs is not None:
                cos_mem = (u_t_n.unsqueeze(1) * mem_negs).sum(dim=-1)
                cos_list.append(cos_mem)
                src_list.append('mem')

        # 批内难负
        cos_batch = (u_t_neg * u_v_neg_batch).sum(dim=-1)
        if neg_source in ('batch', 'hybrid'):
            cos_list.append(cos_batch)
            src_list.append('batch')

        # 合并
        if len(cos_list) == 0:
            cos_neg = cos_batch
        else:
            cos_neg = torch.cat(cos_list, dim=1)
        neg_cos_mean = cos_neg.mean()

        if neg_margin_rad is None:
            neg_margin_rad = 100.0 * math.pi / 180.0
        cos_neg_thresh = math.cos(math.pi - neg_margin_rad)

        if loss_type == 'hinge':
            if use_hard_neg and hard_neg_k is not None and hard_neg_k > 0:
                k = min(hard_neg_k, actual_num_neg)
                topk_vals, _ = torch.topk(cos_neg, k=k, dim=1, largest=True, sorted=False)
                neg_loss = torch.relu(topk_vals - _ensure_scalar(cos_neg_thresh, device, topk_vals.dtype)).mean()
            else:
                neg_loss = torch.relu(cos_neg - _ensure_scalar(cos_neg_thresh, device, cos_neg.dtype)).mean()
        else:
            target_cos = _ensure_scalar(cos_neg_thresh, device, cos_neg.dtype)
            neg_loss = ((cos_neg - target_cos) ** 2).mean()

    loss = pos_loss + neg_loss
    if return_stats:
        stats = {
            'pos_loss': pos_loss.detach(),
            'neg_loss': neg_loss.detach(),
            'pos_cos_mean': cos_pos.mean().detach(),
            'neg_cos_mean': neg_cos_mean.detach(),
        }
        return loss, stats
    return loss


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
    """计算两点的双曲测地距离 d(x,y)（基于Lorentz内积）。
    约定：cosh( sqrt(c) * d ) = - <x,y>_L。
    """
    inner = lorentz_inner(x, y)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
    if c.dim() == 0:
        c = c.view(1)
    # 广播到形状
    while c.dim() < inner.dim():
        c = c.unsqueeze(0)
    sqrt_c = torch.sqrt(torch.clamp(c, min=1e-10))
    arg = torch.clamp(-inner, min=1.0 + 1e-6)
    dist = torch.acosh(arg) / sqrt_c
    return dist


def distance_loss_hyp(
    t_hyp: torch.Tensor,
    v_hyp: torch.Tensor,
    c_t,
    c_v,
    num_neg_samples: int = 32,
    pos_margin: float = 0.5,
    neg_margin: float = 1.5,
    use_hard_neg: bool = True,
    hard_neg_k: int = 8,
    return_stats: bool = False,
    memory_bank: Optional[object] = None,
    neg_source: str = 'hybrid',
    memory_cand_size: int = 4096,
    memory_ratio: float = 0.5,
):
    """
    基于测地距离的hinge损失（并行于切角度）。
    pos: d_pos <= pos_margin
    neg: d_neg >= neg_margin
    """
    device = t_hyp.device
    dtype = t_hyp.dtype
    b = t_hyp.size(0)

    # 正样本距离（使用各自曲率的均值）
    c_mean = (c_t + c_v) / 2.0 if isinstance(c_t, torch.Tensor) else (c_t + c_v) / 2.0
    d_pos = lorentz_distance(t_hyp, v_hyp, c_mean)
    pos_loss = torch.relu(d_pos - pos_margin).mean()

    # 负样本
    neg_loss = torch.tensor(0.0, device=device, dtype=dtype)
    d_neg_mean = torch.tensor(0.0, device=device, dtype=dtype)
    if b > 1:
        actual_num_neg = min(num_neg_samples, b - 1)
        all_idx = torch.arange(b, device=device)
        perm = torch.randperm(b, device=device)[:actual_num_neg]
        neg_idx_mat = perm.unsqueeze(0).expand(b, -1)
        # 防自匹配：如果某行负样本等于自身索引，则用 (i+1)%b 替换同位置
        self_mask = neg_idx_mat == all_idx.unsqueeze(1)
        if self_mask.any():
            replace_vals = ((all_idx + 1) % b).unsqueeze(1).expand_as(neg_idx_mat)
            neg_idx_mat = torch.where(self_mask, replace_vals, neg_idx_mat)

        # 批内候选
        v_neg_batch = v_hyp[neg_idx_mat]
        # 记忆库候选
        d_list = []
        if neg_source in ('memory', 'hybrid') and memory_bank is not None:
            # 用切空间query选记忆库难负，再回到hyp空间计算距离
            # 近似：直接在切空间取mem neg与当前u_v同维度，不做回映射，作为强负的“方向指引”
            pass  # 简化：先使用批内距离，memory用于角度分支即可

        # 计算批内距离
        t_rep = t_hyp.unsqueeze(1).expand(-1, actual_num_neg, -1)
        # 距离需要逐对计算
        d_neg = lorentz_distance(t_rep.reshape(-1, t_hyp.size(-1)), v_neg_batch.reshape(-1, v_hyp.size(-1)), c_mean)
        d_neg = d_neg.view(b, actual_num_neg)
        d_neg_mean = d_neg.mean()
        if use_hard_neg and hard_neg_k > 0:
            k = min(hard_neg_k, actual_num_neg)
            topk_vals, _ = torch.topk(d_neg, k=k, dim=1, largest=False)  # 距离小的是最难
            neg_loss = torch.relu(neg_margin - topk_vals).mean()
        else:
            neg_loss = torch.relu(neg_margin - d_neg).mean()

    loss = pos_loss + neg_loss
    if return_stats:
        stats = {
            'dist_pos_mean': d_pos.mean().detach(),
            'dist_neg_mean': d_neg_mean.detach(),
        }
        return loss, stats
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
    num_neg_samples: int = 32,
    vis_available: torch.Tensor = None,
    txt_available: torch.Tensor = None,
    # 角度损失配置
    loss_type: str = 'hinge',
    pos_margin_rad: float = None,
    neg_margin_rad: float = None,
    use_hard_neg: bool = True,
    hard_neg_k: int = 8,
    return_ang_stats: bool = False,
    # 记忆库/距离并行
    memory_bank: Optional[object] = None,
    neg_source: str = 'hybrid',
    memory_cand_size: int = 4096,
    memory_ratio: float = 0.5,
    lambda_d: float = 0.5,
    use_dist_hinge: bool = True,
    dist_pos_margin: float = 0.5,
    dist_neg_margin: float = 1.5,
) -> dict:
    """
    计算前端总损失，返回各项及可选角度分项统计
    """
    device = t_sem.device

    # 模态缺失：只对同时有文本与视觉的样本计算
    if vis_available is not None and txt_available is not None:
        both_available = vis_available & txt_available
        if not both_available.any():
            ret = {
                'L_front': torch.tensor(0.0, device=device),
                'L_sem': torch.tensor(0.0, device=device),
                'L_ang': torch.tensor(0.0, device=device),
                'L_rad': torch.tensor(0.0, device=device),
            }
            if return_ang_stats:
                ret['ang_stats'] = {
                    'pos_loss': torch.tensor(0.0, device=device),
                    'neg_loss': torch.tensor(0.0, device=device),
                    'pos_cos_mean': torch.tensor(0.0, device=device),
                    'neg_cos_mean': torch.tensor(0.0, device=device),
                }
            return ret

        t_sem_valid = t_sem[both_available]
        v_sem_valid = v_sem[both_available]
        t_hyp_valid = t_hyp[both_available]
        v_hyp_valid = v_hyp[both_available]
        c_t_valid = c_t if not isinstance(c_t, torch.Tensor) or c_t.dim() == 0 else c_t[both_available]
        c_v_valid = c_v if not isinstance(c_v, torch.Tensor) or c_v.dim() == 0 else c_v[both_available]

        L_sem = info_nce_t2v_sem(t_sem_valid, v_sem_valid, temperature=tau, negative_samples=num_neg_samples)

        # 在切空间计算角度
        u_t = log0_lorentz(t_hyp_valid, c_t_valid)
        u_v = log0_lorentz(v_hyp_valid, c_v_valid)
        if return_ang_stats:
            L_ang, ang_stats = angle_loss_tangent(
                u_t, u_v,
                num_neg_samples=num_neg_samples,
                loss_type=loss_type,
                pos_margin_rad=pos_margin_rad,
                neg_margin_rad=neg_margin_rad,
                use_hard_neg=use_hard_neg,
                hard_neg_k=hard_neg_k,
                return_stats=True,
                memory_bank=memory_bank,
                neg_source=neg_source,
                memory_cand_size=memory_cand_size,
                memory_ratio=memory_ratio,
            )
        else:
            L_ang = angle_loss_tangent(
                u_t, u_v,
                num_neg_samples=num_neg_samples,
                loss_type=loss_type,
                pos_margin_rad=pos_margin_rad,
                neg_margin_rad=neg_margin_rad,
                use_hard_neg=use_hard_neg,
                hard_neg_k=hard_neg_k,
                return_stats=False,
                memory_bank=memory_bank,
                neg_source=neg_source,
                memory_cand_size=memory_cand_size,
                memory_ratio=memory_ratio,
            )
        L_rad = radius_reg_hyp(t_hyp_valid, v_hyp_valid, c_t_valid, c_v_valid, r_t_target=r_t_target, r_v_target=r_v_target)
        if use_dist_hinge:
            L_dist = distance_loss_hyp(
                t_hyp_valid, v_hyp_valid, c_t_valid, c_v_valid,
                num_neg_samples=num_neg_samples,
                pos_margin=dist_pos_margin,
                neg_margin=dist_neg_margin,
                use_hard_neg=use_hard_neg,
                hard_neg_k=hard_neg_k,
                return_stats=False,
                memory_bank=memory_bank,
                neg_source=neg_source,
                memory_cand_size=memory_cand_size,
                memory_ratio=memory_ratio,
            )
        else:
            L_dist = torch.tensor(0.0, device=t_sem.device)
    else:
        L_sem = info_nce_t2v_sem(t_sem, v_sem, temperature=tau, negative_samples=num_neg_samples)
        # 在切空间计算角度
        u_t = log0_lorentz(t_hyp, c_t)
        u_v = log0_lorentz(v_hyp, c_v)
        if return_ang_stats:
            L_ang, ang_stats = angle_loss_tangent(
                u_t, u_v,
                num_neg_samples=num_neg_samples,
                loss_type=loss_type,
                pos_margin_rad=pos_margin_rad,
                neg_margin_rad=neg_margin_rad,
                use_hard_neg=use_hard_neg,
                hard_neg_k=hard_neg_k,
                return_stats=True,
                memory_bank=memory_bank,
                neg_source=neg_source,
                memory_cand_size=memory_cand_size,
                memory_ratio=memory_ratio,
            )
        else:
            L_ang = angle_loss_tangent(
                u_t, u_v,
                num_neg_samples=num_neg_samples,
                loss_type=loss_type,
                pos_margin_rad=pos_margin_rad,
                neg_margin_rad=neg_margin_rad,
                use_hard_neg=use_hard_neg,
                hard_neg_k=hard_neg_k,
                return_stats=False,
                memory_bank=memory_bank,
                neg_source=neg_source,
                memory_cand_size=memory_cand_size,
                memory_ratio=memory_ratio,
            )
        L_rad = radius_reg_hyp(t_hyp, v_hyp, c_t, c_v, r_t_target=r_t_target, r_v_target=r_v_target)
        if use_dist_hinge:
            L_dist = distance_loss_hyp(
                t_hyp, v_hyp, c_t, c_v,
                num_neg_samples=num_neg_samples,
                pos_margin=dist_pos_margin,
                neg_margin=dist_neg_margin,
                use_hard_neg=use_hard_neg,
                hard_neg_k=hard_neg_k,
                return_stats=False,
                memory_bank=memory_bank,
                neg_source=neg_source,
                memory_cand_size=memory_cand_size,
                memory_ratio=memory_ratio,
            )
        else:
            L_dist = torch.tensor(0.0, device=t_sem.device)

    L_front = L_sem + lambda_h * (L_ang + lambda_d * L_dist + lambda_r * L_rad)

    ret = {
        'L_sem': L_sem,
        'L_ang': L_ang,
        'L_dist': L_dist,
        'L_rad': L_rad,
        'L_front': L_front,
    }
    if return_ang_stats:
        ret['ang_stats'] = ang_stats
    return ret
