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


@torch.no_grad()
def signal_distribution_concentration(scores: torch.Tensor, prefix: str = "") -> dict:
    """Summarize concentration of a non-negative token score distribution.

    Low entropy/effective support and high top-k mass indicate that the selector
    sees a small set of dominant visual tokens. That should be more directly
    relevant to pruning risk than global feature rank alone.
    """
    x = scores.detach().float().flatten().clamp_min(0)
    n = int(x.numel())
    if n == 0:
        return {}
    total = x.sum()
    if not torch.isfinite(total) or total <= 0:
        p = torch.full_like(x, 1.0 / n)
    else:
        p = x / total

    p_sorted = torch.sort(p, descending=True).values
    entropy = -(p * (p + 1e-12).log()).sum()
    entropy_norm = entropy / torch.log(torch.tensor(float(n), device=p.device))
    effective_support = torch.exp(entropy) / n

    ascending = torch.sort(p).values
    index = torch.arange(1, n + 1, device=p.device, dtype=p.dtype)
    # Gini coefficient for a probability vector; 0=uniform, high=concentrated.
    gini = (2 * (index * ascending).sum() / n) - (n + 1) / n

    def top_mass(k: int) -> float:
        return p_sorted[: min(k, n)].sum().item()

    result = {
        "entropy_norm": entropy_norm.item(),
        "effective_support_norm": effective_support.item(),
        "gini": gini.item(),
        "top1_mass": top_mass(1),
        "top5_mass": top_mass(5),
        "top10_mass": top_mass(10),
        "top25_mass": top_mass(25),
        "max_score": x.max().item(),
        "mean_score": x.mean().item(),
    }
    return {f"{prefix}{k}": v for k, v in result.items()}


@torch.no_grad()
def signal_budget_boundaries(scores: torch.Tensor, ratios, prefix: str = "") -> dict:
    """Selector mass and boundary margin at each candidate keep ratio."""
    x = scores.detach().float().flatten().clamp_min(0)
    n = int(x.numel())
    if n == 0:
        return {}
    total = x.sum()
    p = x / total if torch.isfinite(total) and total > 0 else torch.full_like(x, 1.0 / n)
    ordered = torch.sort(p, descending=True).values
    result = {}
    for ratio in ratios:
        keep = max(1, min(n, int(n * float(ratio))))
        threshold = ordered[keep - 1]
        next_score = ordered[keep] if keep < n else torch.zeros_like(threshold)
        gap = threshold - next_score
        name = str(int(round(float(ratio) * 100)))
        result[f"keep{name}_mass"] = ordered[:keep].sum().item()
        result[f"keep{name}_boundary_gap"] = gap.item()
        result[f"keep{name}_relative_gap"] = (gap / (threshold + 1e-12)).item()
    return {f"{prefix}{key}": value for key, value in result.items()}


@torch.no_grad()
def signal_pair_stability(scores_a: torch.Tensor, scores_b: torch.Tensor,
                          keep_ratio: float = 0.25, prefix: str = "") -> dict:
    """Measure how stable a visual-token ranking is between two layers."""
    a = scores_a.detach().float().flatten().clamp_min(0)
    b = scores_b.detach().float().flatten().clamp_min(0)
    if a.numel() == 0 or a.shape != b.shape:
        return {}

    def normalize(values):
        total = values.sum()
        return values / total if torch.isfinite(total) and total > 0 else torch.full_like(
            values, 1.0 / values.numel()
        )

    pa, pb = normalize(a), normalize(b)
    keep = max(1, min(a.numel(), int(a.numel() * float(keep_ratio))))
    top_a = set(torch.topk(pa, keep).indices.cpu().tolist())
    top_b = set(torch.topk(pb, keep).indices.cpu().tolist())
    union = top_a | top_b
    result = {
        "cosine": torch.nn.functional.cosine_similarity(pa, pb, dim=0).item(),
        "l1": torch.abs(pa - pb).sum().item(),
        "topk_jaccard": len(top_a & top_b) / len(union) if union else 1.0,
    }
    return {f"{prefix}{key}": value for key, value in result.items()}
