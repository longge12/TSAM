"""
双曲几何工具模块
实现 Lorentz 模型的指数映射、对数映射、内积、角度、半径等操作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def softplus_pos(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """确保参数为正数的 softplus 函数"""
    return F.softplus(x, beta=beta) + 1e-6


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算 Lorentz 内积: -x_0*y_0 + sum(x_i*y_i)
    
    Args:
        x: (..., d+1) Lorentz 空间中的向量
        y: (..., d+1) Lorentz 空间中的向量
    
    Returns:
        (...,) 内积标量
    """
    # 第一个分量是时间分量，其余是空间分量
    time_part = -x[..., 0] * y[..., 0]
    space_part = torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    return time_part + space_part


def lorentz_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """计算 Lorentz 范数的平方"""
    return lorentz_inner(x, x)


def exp0_lorentz(u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    从原点出发的 Lorentz 指数映射
    
    Args:
        u: (..., d) 切空间中的向量（欧式空间）
        c: (...,) 或标量，曲率参数
    
    Returns:
        (..., d+1) Lorentz 空间中的向量
    """
    # 确保数值稳定
    u_norm_sq = torch.sum(u ** 2, dim=-1, keepdim=True)
    u_norm = torch.sqrt(torch.clamp(u_norm_sq, min=1e-10))
    c = c if isinstance(c, torch.Tensor) else torch.tensor(c, device=u.device, dtype=u.dtype)
    if c.dim() == 0:
        c = c.unsqueeze(-1)
    while c.dim() < u_norm.dim():
        c = c.unsqueeze(-1)
    
    sqrt_c = torch.sqrt(torch.clamp(c, min=1e-10))
    sqrt_c_inv = 1.0 / sqrt_c
    
    # 避免除零
    u_norm_safe = torch.clamp(u_norm, min=1e-8)
    
    # 计算双曲函数
    cosh_term = torch.cosh(sqrt_c * u_norm_safe)
    sinh_term = torch.sinh(sqrt_c * u_norm_safe)
    
    # 构建 Lorentz 向量: [cosh(||u||_c), sinh(||u||_c) * u / (||u|| * sqrt(c))]
    x0 = cosh_term  # 时间分量
    x_space = (sinh_term * sqrt_c_inv) * (u / u_norm_safe)
    
    # 处理零向量情况
    mask = u_norm.squeeze(-1) < 1e-8
    if mask.any():
        # 零向量映射到原点 [1, 0, 0, ...]
        x_space[mask] = torch.zeros_like(x_space[mask])
        x0[mask] = torch.ones_like(x0[mask])
    
    x = torch.cat([x0, x_space], dim=-1)
    return x


def log0_lorentz(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    到原点的 Lorentz 对数映射
    
    Args:
        x: (..., d+1) Lorentz 空间中的向量
        c: (...,) 或标量，曲率参数
    
    Returns:
        (..., d) 切空间中的向量（欧式空间）
    """
    c = c if isinstance(c, torch.Tensor) else torch.tensor(c, device=x.device, dtype=x.dtype)
    if c.dim() == 0:
        c = c.unsqueeze(-1)
    while c.dim() < x.dim() - 1:
        c = c.unsqueeze(-1)
    
    sqrt_c = torch.sqrt(torch.clamp(c, min=1e-10))
    sqrt_c_inv = 1.0 / sqrt_c
    
    # 提取时间分量和空间分量
    x0 = x[..., 0:1]  # 时间分量
    x_space = x[..., 1:]  # 空间分量
    
    # 计算范数
    x_space_norm_sq = torch.sum(x_space ** 2, dim=-1, keepdim=True)
    x_space_norm = torch.sqrt(torch.clamp(x_space_norm_sq, min=1e-10))
    
    # 计算双曲距离
    # arccosh(x0) = log(x0 + sqrt(x0^2 - 1))
    x0_sq = x0 ** 2
    dist = torch.acosh(torch.clamp(x0, min=1.0 + 1e-6))
    
    # 计算对数映射
    dist_safe = torch.clamp(dist, min=1e-8)
    u = (dist_safe * sqrt_c_inv) * (x_space / x_space_norm)
    
    # 处理原点情况（x0接近1，x_space接近0）
    mask = (x_space_norm.squeeze(-1) < 1e-8) & ((x0.squeeze(-1) - 1.0).abs() < 1e-6)
    if mask.any():
        u[mask] = torch.zeros_like(u[mask])
    
    return u


def lorentz_angle(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    计算 Lorentz 空间中两个向量的角度
    
    Args:
        x: (..., d+1) Lorentz 空间向量
        y: (..., d+1) Lorentz 空间向量
        c: (...,) 或标量，曲率参数
    
    Returns:
        (...,) 角度（弧度）
    """
    # 计算 Lorentz 内积
    inner = lorentz_inner(x, y)
    
    # 计算范数
    x_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(x), min=1e-10))
    y_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(y), min=1e-10))
    
    # 计算余弦值
    # 关键修复：Lorentz空间角度公式需要负号！
    # cos(α) = -⟨x, y⟩_H / (||x||_H * ||y||_H)
    # 这是因为Lorentz内积对于流形上的点通常是负的
    cos_angle = torch.clamp(-inner / (x_norm * y_norm), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    
    # 计算角度
    angle = torch.acos(cos_angle)
    
    return angle


def lorentz_radius(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    计算 Lorentz 空间中向量的半径（到原点的距离）
    
    Args:
        x: (..., d+1) Lorentz 空间向量
        c: (...,) 或标量，曲率参数
    
    Returns:
        (...,) 半径
    """
    # 时间分量即为 cosh(distance * sqrt(c))
    x0 = x[..., 0]
    
    # 计算距离
    # arccosh(x0) = distance * sqrt(c)
    # distance = arccosh(x0) / sqrt(c)
    c = c if isinstance(c, torch.Tensor) else torch.tensor(c, device=x.device, dtype=x.dtype)
    sqrt_c = torch.sqrt(torch.clamp(c, min=1e-10))
    
    radius = torch.acosh(torch.clamp(x0, min=1.0 + 1e-6)) / sqrt_c
    
    return radius


def project_to_lorentz(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    将向量投影到 Lorentz 流形上（确保满足约束条件）
    
    Args:
        x: (..., d+1) 待投影的向量
        c: (...,) 或标量，曲率参数
    
    Returns:
        (..., d+1) 投影后的向量
    """
    # Lorentz 约束: -x_0^2 + sum(x_i^2) = -1/c
    x0 = x[..., 0:1]
    x_space = x[..., 1:]
    
    space_norm_sq = torch.sum(x_space ** 2, dim=-1, keepdim=True)
    c = c if isinstance(c, torch.Tensor) else torch.tensor(c, device=x.device, dtype=x.dtype)
    if c.dim() == 0:
        c = c.unsqueeze(-1)
    while c.dim() < x0.dim():
        c = c.unsqueeze(-1)
    
    # 计算满足约束的时间分量
    c_inv = 1.0 / torch.clamp(c, min=1e-10)
    x0_proj = torch.sqrt(torch.clamp(space_norm_sq + c_inv, min=1e-10))
    
    # 保持符号
    x0_proj = x0_proj * torch.sign(x0)
    
    x_proj = torch.cat([x0_proj, x_space], dim=-1)
    return x_proj

