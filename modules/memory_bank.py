import torch
from typing import Union


class MemoryBank:
    """
    简单的环形记忆库，用于跨批难负样本挖掘。
    存储切空间的向量（已L2归一化），分别维护文本和视觉两套，以便不同方向检索。
    """

    def __init__(self, dim: int, size: int = 65536, device: Union[torch.device, str] = 'cpu'):
        self.size = int(size)
        self.dim = int(dim)
        self.device = torch.device(device)
        self.ptr = 0
        self.full = False

        self.bank_t = torch.zeros(self.size, self.dim, device=self.device)
        self.bank_v = torch.zeros(self.size, self.dim, device=self.device)

    @torch.no_grad()
    def enqueue(self, u_t: torch.Tensor, u_v: torch.Tensor):
        """将一批向量压入记忆库（自动L2归一化）。"""
        if u_t.numel() == 0 or u_v.numel() == 0:
            return
        b = u_t.size(0)
        u_t_n = torch.nn.functional.normalize(u_t.detach(), p=2, dim=-1)
        u_v_n = torch.nn.functional.normalize(u_v.detach(), p=2, dim=-1)

        if b >= self.size:
            # 只保留最后 size 个
            u_t_n = u_t_n[-self.size:]
            u_v_n = u_v_n[-self.size:]
            b = self.size

        end = self.ptr + b
        if end <= self.size:
            self.bank_t[self.ptr:end] = u_t_n
            self.bank_v[self.ptr:end] = u_v_n
        else:
            first = self.size - self.ptr
            second = end - self.size
            self.bank_t[self.ptr:] = u_t_n[:first]
            self.bank_v[self.ptr:] = u_v_n[:first]
            self.bank_t[:second] = u_t_n[first:]
            self.bank_v[:second] = u_v_n[first:]

        self.ptr = (self.ptr + b) % self.size
        if not self.full and self.ptr == 0:
            self.full = True

    @torch.no_grad()
    def sample_hard_neg(self, query: torch.Tensor, from_mod: str = 'v', topk: int = 8, cand_size: int = 4096) -> torch.Tensor:
        """
        从记忆库中为每个query挑选top‑k难负（cos最大）。
        from_mod: 't' | 'v'，表示从文本/视觉库中取候选。
        返回：(batch, topk, dim) 的负样本矩阵。
        """
        if topk <= 0 or self.size == 0:
            return None
        bank = self.bank_v if from_mod == 'v' else self.bank_t
        valid_len = self.size if self.full else self.ptr
        if valid_len <= 0:
            return None

        pool = bank[:valid_len]
        b = query.size(0)
        q = torch.nn.functional.normalize(query, p=2, dim=-1)

        # 随机候选子集，避免全库矩阵乘爆显存
        if cand_size < valid_len:
            idx = torch.randint(0, valid_len, (cand_size,), device=pool.device)
            cand = pool[idx]
        else:
            cand = pool
            idx = None

        # cos 相似度
        sim = q @ cand.t()  # (b, C)
        k = min(topk, sim.size(1))
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        # 取向量
        negs = cand[topk_idx]
        return negs
