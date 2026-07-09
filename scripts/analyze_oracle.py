"""Budget-response oracle analysis.

用法（拿到朋友发回的 tar 包后）:
  tar xzf oracle_results_v2.tar.gz
  python scripts/analyze_oracle.py --dir results/

兼容旧版 oracle 结果；新版结果会额外分析 true full_generate、
preservation oracle、correction oracle 和 non-monotonic 样本。
"""
import argparse
import glob
import json
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIOS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]


def load_records(result_dir):
    files = sorted(
        f for f in glob.glob(f"{result_dir}/oracle*.shard*.json")
        if ".partial." not in f
    )
    records = []
    for f in files:
        with open(f) as fh:
            records += json.load(fh)
    return records, files


def interp_fixed_acc(fixed_acc, budget):
    xs = np.array(RATIOS)
    ys = np.array([fixed_acc[r] for r in RATIOS])
    return float(np.interp(budget, xs, ys))


def bootstrap_ci(values, iters=1000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(iters):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return tuple(np.percentile(means, [2.5, 97.5]))


def record_vals(x):
    return [bool(x["correct"][str(r)]) for r in RATIOS]


def is_full_correct(x):
    return bool(x.get("full_correct", x["correct"]["1.0"]))


def min_ok_budget(vals, default=1.0):
    ok = [r for r, v in zip(RATIOS, vals) if v]
    return min(ok) if ok else default


def is_nonmonotonic(vals):
    return any(v and not all(vals[i:]) for i, v in enumerate(vals[:-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    records, files = load_records(args.dir)
    n = len(records)
    print(f"共 {n} 样本，来自 {len(files)} 个 shard\n")
    if n == 0:
        raise SystemExit("没有找到 oracle*.shard*.json")

    fixed_acc = {r: np.mean([x["correct"][str(r)] for x in records]) for r in RATIOS}
    for r in RATIOS:
        lo, hi = bootstrap_ci(
            [x["correct"][str(r)] for x in records], args.bootstrap, seed=int(r * 10000)
        )
        print(f"fixed {r:>5.2f}: acc {fixed_acc[r]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    true_full = np.mean([is_full_correct(x) for x in records])
    print(f"\ntrue full_generate acc: {true_full:.4f}")

    task_budgets = []
    task_budgets_correct_only = []
    task_correct_flags = []
    preservation_budgets = []
    full_correct_flags = []
    correction_flags = []
    nonmono_flags = []
    prune1_match_flags = []
    pattern_counter = Counter()

    for x in records:
        vals = record_vals(x)
        pattern_counter[tuple(vals)] += 1
        full_ok = is_full_correct(x)
        task_ok = any(vals)

        task_correct_flags.append(task_ok)
        task_budgets.append(min_ok_budget(vals))
        if task_ok:
            task_budgets_correct_only.append(min_ok_budget(vals))
        full_correct_flags.append(full_ok)
        nonmono_flags.append(is_nonmonotonic(vals))

        if full_ok:
            preservation_budgets.append(min_ok_budget(vals))
        correction_flags.append((not full_ok) and any(vals[:-1]))

        if "prune1_matches_full" in x:
            prune1_match_flags.append(bool(x["prune1_matches_full"]))

    task_acc = float(np.mean(task_correct_flags))
    task_budget = float(np.mean(task_budgets))
    fixed_at_task_budget = interp_fixed_acc(fixed_acc, task_budget)
    print(f"\ntask oracle: acc {task_acc:.4f} @ avg budget {task_budget:.3f}")
    print(f"同预算 fixed 插值 acc: {fixed_at_task_budget:.4f}")
    print(f"task oracle gain: {(task_acc - fixed_at_task_budget) * 100:.2f} 个百分点")

    if preservation_budgets:
        preservation_budget = float(np.mean(preservation_budgets))
        preserve_at_10 = np.mean([b <= 0.10 for b in preservation_budgets])
        preserve_at_25 = np.mean([b <= 0.25 for b in preservation_budgets])
        print(
            "\npreservation oracle: "
            f"{len(preservation_budgets)}/{sum(full_correct_flags)} full-correct 样本"
        )
        print(f"avg min budget among full-correct: {preservation_budget:.3f}")
        print(f"full-correct samples preservable <=10%: {preserve_at_10:.4f}")
        print(f"full-correct samples preservable <=25%: {preserve_at_25:.4f}")

    correction_rate = float(np.mean(correction_flags))
    nonmono_rate = float(np.mean(nonmono_flags))
    print(f"\ncorrection oracle samples: {sum(correction_flags)}/{n} ({correction_rate:.4f})")
    print(f"non-monotonic samples: {sum(nonmono_flags)}/{n} ({nonmono_rate:.4f})")
    if prune1_match_flags:
        print(
            "prune_generate(1.0) == full_generate text match: "
            f"{np.mean(prune1_match_flags):.4f}"
        )

    print("\nmin correct budget distribution:")
    for budget, count in sorted(Counter(task_budgets_correct_only).items()):
        print(f"  {budget:>4}: {count}")
    print(f"  no correct: {n - len(task_budgets_correct_only)}")

    print("\ntop budget-response patterns:")
    for pat, count in pattern_counter.most_common(10):
        print(f"  {count:>5}: {pat}")

    verdict = (
        "GO: oracle 空间很大，继续做 risk-aware router"
        if task_acc - fixed_at_task_budget >= 0.05
        else "CAUTION: 只能作为现象分析，先验证 router"
        if task_acc - fixed_at_task_budget >= 0.02
        else "NO-GO: oracle 空间不足"
    )
    print(f"\n=== 判定: {verdict} ===")

    xs = np.array(RATIOS)
    ys = np.array([fixed_acc[r] for r in RATIOS])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xs, ys, "o-", label="Fixed ratio")
    axes[0].scatter([task_budget], [task_acc], marker="*", s=250, c="red",
                    zorder=5, label="Task oracle")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Avg token retention ratio")
    axes[0].set_ylabel("GQA accuracy")
    axes[0].legend()
    axes[0].set_title("Budget curve")

    axes[1].hist(task_budgets, bins=RATIOS + [1.5], edgecolor="k")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Min correct budget")
    axes[1].set_ylabel("#samples")
    axes[1].set_title("Task oracle budget need")

    if preservation_budgets:
        axes[2].hist(preservation_budgets, bins=RATIOS + [1.5], edgecolor="k")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("Min preserving budget")
        axes[2].set_title("Preservation budget")
    else:
        axes[2].axis("off")

    fig.tight_layout()
    out = f"{args.dir}/oracle_analysis_v2.png"
    fig.savefig(out, dpi=150)
    print(f"\n图已存: {out}")


if __name__ == "__main__":
    main()
