# 前端双分支架构实现说明

## 概述

本次实现在前端新增了双分支架构：语义分支（欧式拉近）+ 几何分支（双曲拉远），并通过门控融合将两者结合。该架构在保持原有编码器（Text/Vision）实现不变的前提下，实现了模态对齐的增强。

## 架构设计

### 1. 编码器层（保持不变）
- Text 编码器：BERT/DeBERTa → `e_t ∈ R^d`
- Vision 编码器：BEiT/ViT → `e_v ∈ R^d`

### 2. 双分支头（新增，参数独立）

#### 语义分支头（SemanticHead）
- **位置**: `models/heads.py`
- **功能**: 欧式空间对齐，拉近同一实体的 t-v 对
- **实现方式**: 
  - Cross-Attention 模式（默认）
  - 双塔投影模式（可选）
- **输出**: `t_sem, v_sem ∈ R^d`

#### 几何分支头（HyperbolicHead）
- **位置**: `models/heads.py`
- **功能**: 双曲空间对齐，通过角度对齐和半径分层保持层级差异
- **实现方式**: Lorentz 模型，可学习曲率
- **输出**: `t_hyp, v_hyp ∈ R^(d+1)`（双曲空间）

### 3. 融合模块（GatedFusion）
- **位置**: `modules/fusion.py`
- **功能**: 门控融合语义和几何分支的输出
- **实现方式**: 
  - 将双曲向量映射回切空间（欧式空间）
  - 使用门控网络学习融合权重
  - 支持差异性正则和门控正则
- **输出**: `t_fuse, v_fuse ∈ R^d`

### 4. 损失函数

#### 前端损失（Front Loss）
- **位置**: `losses/alignment_losses.py`
- **组成**:
  - `L_sem`: InfoNCE 损失（语义分支）
  - `L_ang`: 角度损失（几何分支，最小化正样本夹角）
  - `L_rad`: 半径正则（几何分支，分层约束）
  - `L_fuse_reg`: 融合正则（差异性 + 门控正则）

#### 总损失
```
L_total = L_KGC + L_SACL + L_front
```

## 文件结构

```
TSAM-main/
├── utils/
│   ├── __init__.py
│   └── hyperbolic.py          # 双曲几何工具
├── models/
│   ├── __init__.py
│   └── heads.py                # 语义和几何分支头
├── modules/
│   ├── __init__.py
│   └── fusion.py               # 门控融合模块
├── losses/
│   ├── __init__.py
│   └── alignment_losses.py     # 对齐损失函数
├── TSAM.py                     # 主模型（已修改）
└── Train.py                    # 训练脚本（已修改）
```

## 使用方法

### 1. 启用双分支架构

```bash
python Train.py --use_dual_branch --use_cross_attn
```

### 2. 配置参数

```bash
python Train.py \
  --use_dual_branch \
  --lambda_h 1.0 \          # 几何损失权重
  --lambda_r 0.1 \          # 半径正则权重
  --lambda_f 0.1 \          # 融合正则权重
  --tau 0.07 \              # InfoNCE 温度
  --angle_margin 0.1 \      # 角度损失 margin
  --curvature_init 0.1 \    # 曲率初始值
  --use_diversity_reg       # 启用差异性正则
```

### 3. 禁用双分支（使用原始架构）

```bash
python Train.py  # 默认启用，可通过 --no-use_dual_branch 禁用
```

## 关键特性

1. **向后兼容**: 通过 `use_dual_branch` 参数控制，不影响原有代码
2. **模块化设计**: 各模块独立，易于扩展和维护
3. **数值稳定**: 双曲几何操作包含数值稳定性处理
4. **可配置**: 支持多种融合策略和损失权重配置

## 训练日志

启用双分支后，训练日志会显示：
- Total: 总损失
- KGC: KGC 损失
- SACL: 结构对齐损失
- Front: 前端总损失
- Sem: 语义分支损失
- Ang: 角度损失
- Rad: 半径正则损失
- FuseReg: 融合正则损失

## 中间结果

在训练时，可以通过 `return_intermediates=True` 获取中间结果：
- `t_sem, v_sem`: 语义分支输出
- `t_hyp, v_hyp`: 几何分支输出（双曲空间）
- `t_fuse, v_fuse`: 融合后的输出
- `c_t, c_v`: 曲率参数
- `gate_info`: 门控权重信息

## 注意事项

1. 双分支架构会增加模型参数量和计算开销
2. 建议根据数据集大小调整损失权重（`lambda_h`, `lambda_r`, `lambda_f`）
3. 曲率参数会自动学习，初始值建议设置为 0.1
4. 门控融合的权重分布可以用于分析样本难度

## 后续扩展

预留接口（未实现）：
- 方案 B: 一致性 + 线性融合
- 方案 C: 对比式跨视角融合
- PCGrad/MGDA 梯度手术

可通过修改 `modules/fusion.py` 和 `Train.py` 实现。

