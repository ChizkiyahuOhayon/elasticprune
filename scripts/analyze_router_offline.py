"""Offline router feasibility analysis from saved oracle records.

This script does not run a model. It consumes v2 oracle JSON files produced by
scripts/oracle_gqa.py and evaluates whether cheap signals or simple router rules
can predict useful per-sample token budgets.

Example:
  python scripts/analyze_router_offline.py \
    --dir ../results_v2_complete \
    --out results/router_offline.json \
    --csv results/router_labeled_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elasticprune.eval_utils import (  # noqa: E402
    RATIOS,
    bootstrap_ci,
    correct_vector,
    evaluate_budget_assignment,
    fixed_budget,
    full_correct,
    mean,
    min_correct_budget,
    question_type,
    ratio_key,
    token_count,
)
from elasticprune.routers import (  # noqa: E402
    CascadeAgreementRouter,
    PairAgreementRouter,
    QuantileFeatureRouter,
    QuestionTypeRouter,
    RandomAdaptiveRouter,
)


FEATURES = [
    "redundancy_erank",
    "query_specificity_entropy",
    "n_text_tokens",
    "question_len",
]

LABELS = [
    "easy_2pct",
    "hard_preserve_gt_25",
    "fragile",
    "correction",
    "nonmonotonic",
]


def load_records(result_dir: str) -> List[Dict]:
    files = sorted(
        f for f in glob(os.path.join(result_dir, "oracle*.shard*.json"))
        if ".partial." not in f
    )
    records: List[Dict] = []
    for path in files:
        with open(path) as f:
            records.extend(json.load(f))
    if not records:
        raise SystemExit(f"No oracle shard JSON files found under {result_dir}")
    return records


def is_nonmonotonic_from_vector(vals: List[bool]) -> bool:
    return any(good and not all(vals[i:]) for i, good in enumerate(vals[:-1]))


def labeled_rows(records: List[Dict]) -> List[Dict]:
    rows = []
    for record in records:
        vals = correct_vector(record)
        min_budget = min_correct_budget(record)
        is_full_correct = full_correct(record)
        sig = record.get("signals") or {}
        row = {
            "qid": record.get("qid"),
            "imageId": record.get("imageId"),
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
            "question_type": question_type(record.get("question", "")),
            "question_len": token_count(record.get("question", "")),
            "pattern": "".join("1" if x else "0" for x in vals),
            "full_correct": is_full_correct,
            "min_correct_budget": min_budget,
            "preserve_budget": min_budget if is_full_correct else None,
            "easy_2pct": is_full_correct and vals[0],
            "hard_preserve_gt_25": is_full_correct and (min_budget is None or min_budget > 0.25),
            "fragile": is_full_correct and any(not good for good in vals[:-1]),
            "correction": (not is_full_correct) and any(vals[:-1]),
            "nonmonotonic": is_nonmonotonic_from_vector(vals),
            "prune1_matches_full": bool(record.get("prune1_matches_full", False)),
            "redundancy_erank": sig.get("redundancy_erank"),
            "query_specificity_entropy": sig.get("query_specificity_entropy"),
            "n_text_tokens": sig.get("n_text_tokens"),
            "_record": record,
        }
        rows.append(row)
    return rows


def finite(values):
    return [float(x) for x in values if x is not None and np.isfinite(float(x))]


def describe(values) -> Dict:
    vals = finite(values)
    if not vals:
        return {"n": 0}
    percentiles = np.percentile(vals, [0, 10, 25, 50, 75, 90, 100])
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "max": float(percentiles[6]),
    }


def spearman(xs, ys) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def roc_auc(scores, labels) -> Optional[float]:
    pairs = [(float(s), bool(y)) for s, y in zip(scores, labels) if s is not None]
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        positives = sum(1 for _, label in sorted_pairs[i:j] if label)
        rank_sum_pos += positives * avg_rank
        i = j
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def evaluate_router(router, records: List[Dict], rows: List[Dict]) -> Dict:
    budgets = router.route(records, rows)
    result = evaluate_budget_assignment(records, budgets)
    cumulative = router.cumulative_costs(records, budgets)
    if cumulative is not None:
        result["avg_cumulative_budget"] = mean(cumulative)
    return result


def summarize_question_types(rows: List[Dict]) -> Dict:
    result = {}
    for qtype in sorted(set(row["question_type"] for row in rows)):
        subset = [row for row in rows if row["question_type"] == qtype]
        result[qtype] = {
            "n": len(subset),
            "full_acc": mean(float(row["full_correct"]) for row in subset),
            "easy_2pct_rate": mean(float(row["easy_2pct"]) for row in subset),
            "fragile_rate": mean(float(row["fragile"]) for row in subset),
            "correction_rate": mean(float(row["correction"]) for row in subset),
            "nonmonotonic_rate": mean(float(row["nonmonotonic"]) for row in subset),
            "avg_min_correct_budget": mean(
                row["min_correct_budget"] if row["min_correct_budget"] is not None else 1.0
                for row in subset
            ),
        }
    return result


def build_summary(records: List[Dict], rows: List[Dict], target_budgets: List[float], bootstrap: int) -> Dict:
    preserve_rows = [row for row in rows if row["preserve_budget"] is not None]
    label_summary = {
        label: {
            "positive": sum(1 for row in rows if row[label]),
            "rate": mean(float(row[label]) for row in rows),
        }
        for label in LABELS
    }

    feature_summary = {feature: describe(row[feature] for row in rows) for feature in FEATURES}
    preserve_spearman = {
        feature: spearman(
            [row[feature] for row in preserve_rows],
            [row["preserve_budget"] for row in preserve_rows],
        )
        for feature in FEATURES
    }

    label_predictiveness = {}
    for label in LABELS:
        label_predictiveness[label] = {}
        for feature in FEATURES:
            auc = roc_auc([row[feature] for row in rows], [row[label] for row in rows])
            pos = [row[feature] for row in rows if row[label]]
            neg = [row[feature] for row in rows if not row[label]]
            label_predictiveness[label][feature] = {
                "pos_mean": mean(pos),
                "neg_mean": mean(neg),
                "auc_high_positive": auc,
                "auc_low_positive": None if auc is None else 1.0 - auc,
            }

    fixed = {
        ratio_key(ratio): {
            **fixed_budget(records, ratio),
            "acc_ci95": bootstrap_ci(
                [float(record["correct"][ratio_key(ratio)]) for record in records],
                iters=bootstrap,
                seed=int(ratio * 10000),
            ),
        }
        for ratio in RATIOS
    }

    routers = {}
    for target in target_budgets:
        key = ratio_key(target)
        routers[key] = {}
        random_results = [
            evaluate_router(RandomAdaptiveRouter(target, seed=seed), records, rows)
            for seed in range(20)
        ]
        routers[key]["random_adaptive"] = {
            "acc": mean(result["acc"] for result in random_results),
            "acc_std": float(np.std([result["acc"] for result in random_results])),
            "avg_budget": mean(result["avg_budget"] for result in random_results),
        }
        for feature in FEATURES:
            for higher_is_harder in [True, False]:
                name = f"{feature}_{'high' if higher_is_harder else 'low'}_hard"
                router = QuantileFeatureRouter(feature, target, higher_is_harder)
                routers[key][name] = evaluate_router(router, records, rows)

    heuristic_routers = {
        "yes_no_0.02_else_0.25": evaluate_router(
            QuestionTypeRouter({"yes_no": 0.02}, default_budget=0.25),
            records,
            rows,
        ),
        "yes_no_0.02_color_0.10_else_0.25": evaluate_router(
            QuestionTypeRouter({"yes_no": 0.02, "color": 0.10}, default_budget=0.25),
            records,
            rows,
        ),
        "yes_no_0.05_color_0.10_what_0.25_else_0.50": evaluate_router(
            QuestionTypeRouter(
                {"yes_no": 0.05, "color": 0.10, "what": 0.25},
                default_budget=0.50,
            ),
            records,
            rows,
        ),
    }

    stability_routers = {
        "pair_0.02_0.05_else_0.25": evaluate_router(
            PairAgreementRouter(0.02, 0.05, 0.25), records, rows
        ),
        "pair_0.02_0.10_else_0.25": evaluate_router(
            PairAgreementRouter(0.02, 0.10, 0.25), records, rows
        ),
        "pair_0.02_0.10_else_0.50": evaluate_router(
            PairAgreementRouter(0.02, 0.10, 0.50), records, rows
        ),
        "cascade_0.02_0.10_0.25_0.50": evaluate_router(
            CascadeAgreementRouter([0.02, 0.10, 0.25, 0.50]), records, rows
        ),
        "cascade_0.02_0.10_0.25_0.50_1.0": evaluate_router(
            CascadeAgreementRouter([0.02, 0.10, 0.25, 0.50, 1.00]), records, rows
        ),
    }

    return {
        "n_samples": len(records),
        "features": FEATURES,
        "labels": LABELS,
        "label_summary": label_summary,
        "feature_summary": feature_summary,
        "preserve_budget_spearman": preserve_spearman,
        "label_predictiveness": label_predictiveness,
        "question_type_summary": summarize_question_types(rows),
        "fixed": fixed,
        "signal_routers": routers,
        "heuristic_routers": heuristic_routers,
        "stability_routers": stability_routers,
    }


def write_labeled_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = [
        "qid", "imageId", "question", "answer", "question_type", "question_len",
        "pattern", "full_correct", "min_correct_budget", "preserve_budget",
        "easy_2pct", "hard_preserve_gt_25", "fragile", "correction",
        "nonmonotonic", "prune1_matches_full", *FEATURES,
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path: str, summary: Dict) -> None:
    lines = [
        "# Offline Router Analysis",
        "",
        f"Samples: {summary['n_samples']}",
        "",
        "## Preserve-Budget Spearman",
        "",
        "| Feature | Spearman r |",
        "|---|---:|",
    ]
    for feature, value in summary["preserve_budget_spearman"].items():
        lines.append(f"| `{feature}` | {value:.4f} |")
    lines.extend([
        "",
        "## Fixed Baselines",
        "",
        "| Budget | Accuracy | Avg budget |",
        "|---:|---:|---:|",
    ])
    for ratio, result in summary["fixed"].items():
        lines.append(f"| {float(ratio):.2f} | {result['acc']:.4f} | {result['avg_budget']:.3f} |")
    lines.extend([
        "",
        "## Stability Routers",
        "",
        "| Router | Accuracy | Final budget | Cumulative budget |",
        "|---|---:|---:|---:|",
    ])
    for name, result in summary["stability_routers"].items():
        final_budget = result.get("avg_budget", result.get("avg_final_budget", float("nan")))
        cumulative = result.get("avg_cumulative_budget", float("nan"))
        lines.append(f"| `{name}` | {result['acc']:.4f} | {final_budget:.3f} | {cumulative:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing oracle shard JSON files")
    parser.add_argument("--out", default="results/router_offline.json")
    parser.add_argument("--csv", default="results/router_labeled_samples.csv")
    parser.add_argument("--md", default="results/router_offline.md")
    parser.add_argument("--target-budgets", default="0.10,0.25,0.35")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    records = load_records(args.dir)
    rows = labeled_rows(records)
    target_budgets = [float(x) for x in args.target_budgets.split(",") if x.strip()]
    summary = build_summary(records, rows, target_budgets, args.bootstrap)

    for path in [args.out, args.csv, args.md]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_labeled_csv(args.csv, rows)
    write_markdown(args.md, summary)

    print(json.dumps({
        "n_samples": summary["n_samples"],
        "preserve_budget_spearman": summary["preserve_budget_spearman"],
        "fixed": summary["fixed"],
        "heuristic_routers": summary["heuristic_routers"],
        "stability_routers": summary["stability_routers"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
