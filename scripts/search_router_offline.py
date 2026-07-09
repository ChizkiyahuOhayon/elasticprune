"""Train/test offline search for budget routers.

This script uses saved oracle records to ask a stricter question than
analyze_router_offline.py:

  If we choose a router on a calibration split, does it still beat baselines on
  a held-out split?

It does not run a model and it never uses ground-truth labels inside a router at
test time. Ground truth is used only to select/evaluate routers offline.

Example:
  python scripts/search_router_offline.py \
    --dir ../results_v2_complete \
    --out results/router_search.json \
    --md results/router_search.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elasticprune.eval_utils import (  # noqa: E402
    RATIOS,
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
    BudgetRouter,
    CascadeAgreementRouter,
    PairAgreementRouter,
    QuantileFeatureRouter,
    QuestionTypeRouter,
    RandomAdaptiveRouter,
)

FEATURES = ["redundancy_erank", "query_specificity_entropy", "n_text_tokens", "question_len"]


@dataclass(frozen=True)
class Candidate:
    name: str
    router: BudgetRouter
    cost_mode: str = "final"  # "final" or "cumulative"


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


def nonmonotonic(vals: Sequence[bool]) -> bool:
    return any(good and not all(vals[i:]) for i, good in enumerate(vals[:-1]))


def make_rows(records: List[Dict]) -> List[Dict]:
    rows = []
    for record in records:
        vals = correct_vector(record)
        min_budget = min_correct_budget(record)
        is_full_correct = full_correct(record)
        sig = record.get("signals") or {}
        rows.append({
            "qid": record.get("qid"),
            "question_type": question_type(record.get("question", "")),
            "question_len": token_count(record.get("question", "")),
            "full_correct": is_full_correct,
            "min_correct_budget": min_budget,
            "preserve_budget": min_budget if is_full_correct else None,
            "easy_2pct": is_full_correct and vals[0],
            "fragile": is_full_correct and any(not good for good in vals[:-1]),
            "correction": (not is_full_correct) and any(vals[:-1]),
            "nonmonotonic": nonmonotonic(vals),
            "redundancy_erank": sig.get("redundancy_erank"),
            "query_specificity_entropy": sig.get("query_specificity_entropy"),
            "n_text_tokens": sig.get("n_text_tokens"),
        })
    return rows


def split_indices(n: int, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    return idx[:n_train].tolist(), idx[n_train:].tolist()


def take(items: List, indices: List[int]) -> List:
    return [items[i] for i in indices]


def evaluate_candidate(candidate: Candidate, records: List[Dict], rows: List[Dict]) -> Dict:
    budgets = candidate.router.route(records, rows)
    result = evaluate_budget_assignment(records, budgets)
    cumulative = candidate.router.cumulative_costs(records, budgets)
    if cumulative is not None:
        result["avg_cumulative_budget"] = mean(cumulative)
    result["selection_cost"] = (
        result["avg_cumulative_budget"]
        if candidate.cost_mode == "cumulative" and "avg_cumulative_budget" in result
        else result["avg_budget"]
    )
    return result


def fixed_results(records: List[Dict]) -> Dict:
    return {ratio_key(ratio): fixed_budget(records, ratio) for ratio in RATIOS}


def random_results(records: List[Dict], target: float, seeds: int = 20) -> Dict:
    values = [
        evaluate_budget_assignment(records, RandomAdaptiveRouter(target, seed=s).route(records))
        for s in range(seeds)
    ]
    return {
        "acc": mean(v["acc"] for v in values),
        "acc_std": float(np.std([v["acc"] for v in values])),
        "avg_budget": mean(v["avg_budget"] for v in values),
    }


def feature_candidates(targets: Iterable[float]) -> List[Candidate]:
    candidates = []
    for target in targets:
        for feature in FEATURES:
            candidates.append(Candidate(
                f"{feature}_high_hard_target_{ratio_key(target)}",
                QuantileFeatureRouter(feature, target, higher_is_harder=True),
            ))
            candidates.append(Candidate(
                f"{feature}_low_hard_target_{ratio_key(target)}",
                QuantileFeatureRouter(feature, target, higher_is_harder=False),
            ))
    return candidates


def stability_candidates() -> List[Candidate]:
    pairs = [
        (0.02, 0.05, 0.10),
        (0.02, 0.05, 0.25),
        (0.02, 0.10, 0.25),
        (0.02, 0.10, 0.50),
        (0.05, 0.10, 0.25),
        (0.05, 0.25, 0.50),
    ]
    cascades = [
        [0.02, 0.05, 0.10, 0.25],
        [0.02, 0.10, 0.25, 0.50],
        [0.02, 0.10, 0.25, 0.50, 1.00],
    ]
    candidates = [
        Candidate(
            f"pair_{ratio_key(low)}_{ratio_key(high)}_else_{ratio_key(fallback)}",
            PairAgreementRouter(low, high, fallback),
            cost_mode="cumulative",
        )
        for low, high, fallback in pairs
    ]
    candidates.extend(
        Candidate(
            "cascade_" + "_".join(ratio_key(x) for x in stages),
            CascadeAgreementRouter(stages),
            cost_mode="cumulative",
        )
        for stages in cascades
    )
    return candidates


def question_type_candidates(train_records: List[Dict], train_rows: List[Dict], targets: Iterable[float]) -> List[Candidate]:
    qtypes = sorted(set(row["question_type"] for row in train_rows))
    budgets = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    counts = {qtype: 0 for qtype in qtypes}
    correct = {qtype: {budget: 0 for budget in budgets} for qtype in qtypes}
    for record, row in zip(train_records, train_rows):
        qtype = row["question_type"]
        counts[qtype] += 1
        for budget in budgets:
            correct[qtype][budget] += int(record["correct"][ratio_key(budget)])

    candidates = []
    for target in targets:
        best = None
        for combo in itertools.product(budgets, repeat=len(qtypes)):
            mapping = dict(zip(qtypes, combo))
            avg_budget = sum(counts[qtype] * mapping[qtype] for qtype in qtypes) / len(train_rows)
            if avg_budget > target:
                continue
            acc = sum(correct[qtype][mapping[qtype]] for qtype in qtypes) / len(train_rows)
            result = {"acc": acc, "avg_budget": avg_budget, "selection_cost": avg_budget}
            if best is None or acc > best[0]["acc"]:
                best = (result, mapping)
        if best is None:
            continue
        _, mapping = best
        name = "question_type_calibrated_target_" + ratio_key(target)
        candidates.append(Candidate(name, QuestionTypeRouter(mapping, default_budget=0.25, router_name=name)))
    return candidates


def select_best(candidates: List[Candidate], train_records: List[Dict], train_rows: List[Dict], target: float):
    evaluated = []
    for candidate in candidates:
        result = evaluate_candidate(candidate, train_records, train_rows)
        if result["selection_cost"] <= target:
            evaluated.append((candidate, result))
    if not evaluated:
        return None
    evaluated.sort(key=lambda item: (item[1]["acc"], -item[1]["selection_cost"]), reverse=True)
    return evaluated[0]


def serialize_candidate(candidate: Candidate, train_result: Dict, test_result: Dict) -> Dict:
    payload = {
        "name": candidate.name,
        "router_class": candidate.router.__class__.__name__,
        "cost_mode": candidate.cost_mode,
        "train": train_result,
        "test": test_result,
    }
    if isinstance(candidate.router, QuestionTypeRouter):
        payload["mapping"] = dict(candidate.router.mapping)
        payload["default_budget"] = candidate.router.default_budget
    return payload


def run_search(records: List[Dict], rows: List[Dict], targets: List[float], train_frac: float, seed: int) -> Dict:
    train_idx, test_idx = split_indices(len(records), train_frac, seed)
    train_records, test_records = take(records, train_idx), take(records, test_idx)
    train_rows, test_rows = take(rows, train_idx), take(rows, test_idx)

    base_candidates = feature_candidates(targets) + stability_candidates()
    calibrated_qtype = question_type_candidates(train_records, train_rows, targets)
    all_candidates = base_candidates + calibrated_qtype

    selections = {}
    for target in targets:
        selected = select_best(all_candidates, train_records, train_rows, target)
        if selected is None:
            selections[ratio_key(target)] = None
            continue
        candidate, train_result = selected
        test_result = evaluate_candidate(candidate, test_records, test_rows)
        selections[ratio_key(target)] = serialize_candidate(candidate, train_result, test_result)

    return {
        "n_samples": len(records),
        "train_size": len(train_records),
        "test_size": len(test_records),
        "train_frac": train_frac,
        "seed": seed,
        "targets": targets,
        "train_fixed": fixed_results(train_records),
        "test_fixed": fixed_results(test_records),
        "test_random": {
            ratio_key(target): random_results(test_records, target)
            for target in targets
        },
        "selected": selections,
    }


def write_markdown(path: str, summary: Dict) -> None:
    lines = [
        "# Offline Router Search",
        "",
        f"Samples: {summary['n_samples']}  Train: {summary['train_size']}  Test: {summary['test_size']}",
        "",
        "## Selected Routers",
        "",
        "| Target cost | Router | Train acc | Train cost | Test acc | Test cost |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for target, selected in summary["selected"].items():
        if selected is None:
            lines.append(f"| {float(target):.2f} | none | - | - | - | - |")
            continue
        lines.append(
            f"| {float(target):.2f} | `{selected['name']}` | "
            f"{selected['train']['acc']:.4f} | {selected['train']['selection_cost']:.3f} | "
            f"{selected['test']['acc']:.4f} | {selected['test']['selection_cost']:.3f} |"
        )
    lines.extend([
        "",
        "## Test Fixed Baselines",
        "",
        "| Budget | Accuracy |",
        "|---:|---:|",
    ])
    for ratio, result in summary["test_fixed"].items():
        lines.append(f"| {float(ratio):.2f} | {result['acc']:.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", default="results/router_search.json")
    parser.add_argument("--md", default="results/router_search.md")
    parser.add_argument("--targets", default="0.10,0.25,0.35")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_records(args.dir)
    rows = make_rows(records)
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    summary = run_search(records, rows, targets, args.train_frac, args.seed)

    for path in [args.out, args.md]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown(args.md, summary)
    print(json.dumps(summary["selected"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
