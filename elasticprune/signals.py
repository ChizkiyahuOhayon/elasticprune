"""每样本预算信号（全部为推理免费副产品，training-free）。

signal_redundancy: 图像自身可压缩程度（patch 特征有效秩，越低越可压）
signal_specificity: 查询特异性（query 与图像 token 在 projector 空间的对齐锐度，
                    越锐 = 越局部的问题 = 越可激进剪枝）
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def signal_redundancy(image_features: torch.Tensor) -> float:
    """image_features: (N, D) projector 之后的图像 token 特征。
    返回归一化有效秩 in (0, 1]，低 = 冗余高 = 预算可小。"""
    x = F.normalize(image_features.float(), dim=-1)
    # 奇异值熵定义的有效秩 (Roy & Vetterli)
    s = torch.linalg.svdvals(x)
    p = s / s.sum()
    p = p[p > 1e-8]
    erank = torch.exp(-(p * p.log()).sum())
    return (erank / len(s)).item()


@torch.no_grad()
def signal_specificity(image_features: torch.Tensor, query_embeds: torch.Tensor) -> float:
    """image_features: (N, D) projector 后图像 token；query_embeds: (M, D) 问题文本
    的 LLM 输入 embedding（同一空间，无需额外模型）。
    对每个 query token 算其在图像 token 上的注意力分布熵，取最小熵（最局部的词）。
    返回归一化熵 in (0,1]，低 = 问题聚焦局部 = 预算可小。"""
    q = F.normalize(query_embeds.float(), dim=-1)
    v = F.normalize(image_features.float(), dim=-1)
    sim = q @ v.T  # (M, N)
    p = F.softmax(sim / 0.07, dim=-1)
    ent = -(p * (p + 1e-9).log()).sum(-1)  # (M,)
    max_ent = torch.log(torch.tensor(float(v.shape[0])))
    return (ent.min() / max_ent).item()


def difficulty_score(redundancy: float, specificity: float, w: float = 0.5) -> float:
    """合成难度分，高 = 需要更多 token。两个信号都是"低=可激进"，直接加权。"""
    return w * redundancy + (1 - w) * specificity
