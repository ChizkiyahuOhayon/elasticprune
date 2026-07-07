"""Oracle 结果分析 + Go/No-Go 判定。

用法（拿到朋友发回的 tar 包后）:
  tar xzf oracle_results.tar.gz
  python scripts/analyze_oracle.py --dir results/
产出: 悬崖曲线图 + oracle 对比 + go/no-go 结论
"""
import argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIOS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()

    records = []
    for f in sorted(glob.glob(f"{args.dir}/oracle*.shard*.json")):
        records += json.load(open(f))
    print(f"共 {len(records)} 样本，来自 {len(glob.glob(f'{args.dir}/oracle*.shard*.json'))} 个 shard\n")

    # 固定比例曲线
    fixed_acc = {r: np.mean([x["correct"][str(r)] for x in records]) for r in RATIOS}
    for r in RATIOS:
        print(f"fixed {r:>5.2f}: acc {fixed_acc[r]:.4f}")

    # oracle：每样本最小可答对预算（答不对记 1.0）
    budgets = []
    for x in records:
        ok = [r for r in RATIOS if x["correct"][str(r)]]
        budgets.append(min(ok) if ok else 1.0)
    oracle_acc = np.mean([any(x["correct"].values()) for x in records])
    oracle_budget = np.mean(budgets)
    print(f"\noracle: acc {oracle_acc:.4f} @ avg budget {oracle_budget:.3f}")

    # 同预算下固定比例的插值精度
    xs, ys = np.array(RATIOS), np.array([fixed_acc[r] for r in RATIOS])
    fixed_at_oracle_budget = np.interp(oracle_budget, xs, ys)
    gain = oracle_acc - fixed_at_oracle_budget
    print(f"同预算({oracle_budget:.3f})固定比例插值 acc: {fixed_at_oracle_budget:.4f}")
    print(f"oracle 增益: {gain*100:.2f} 个百分点")

    verdict = ("GO (全速推进)" if gain >= 0.05 else
               "GO (悬崖分析为主贡献)" if gain >= 0.02 else
               "NO-GO (止损，切 idea B)")
    print(f"\n=== 判定: {verdict} ===")

    # 图1: 悬崖曲线 + oracle 点
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(xs, ys, "o-", label="Fixed ratio")
    axes[0].scatter([oracle_budget], [oracle_acc], marker="*", s=250,
                    c="red", zorder=5, label="Oracle (per-sample)")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Avg token retention ratio")
    axes[0].set_ylabel("GQA accuracy"); axes[0].legend(); axes[0].set_title("Cliff curve")

    # 图2: 每样本最小预算分布（motivation 核心图）
    axes[1].hist(budgets, bins=RATIOS + [1.5], edgecolor="k")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Min sufficient budget per sample")
    axes[1].set_ylabel("#samples"); axes[1].set_title("Per-sample budget need (heterogeneity)")
    fig.tight_layout()
    fig.savefig(f"{args.dir}/oracle_analysis.png", dpi=150)
    print(f"图已存: {args.dir}/oracle_analysis.png")


if __name__ == "__main__":
    main()
