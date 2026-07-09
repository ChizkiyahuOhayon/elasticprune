"""Utilities for offline evaluation of budget-response oracle records."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np

RATIOS: List[float] = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]


def ratio_key(ratio: float) -> str:
    return str(float(ratio))


def normalize_text(text: object) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_text(answer) in normalize_text(prediction)


def correct_vector(record: Dict) -> List[bool]:
    return [bool(record["correct"][ratio_key(r)]) for r in RATIOS]


def min_correct_budget(record: Dict) -> Optional[float]:
    for ratio, correct in zip(RATIOS, correct_vector(record)):
        if correct:
            return ratio
    return None


def full_correct(record: Dict) -> bool:
    return bool(record.get("full_correct", record["correct"][ratio_key(1.0)]))


def is_nonmonotonic(record: Dict) -> bool:
    vals = correct_vector(record)
    return any(correct and not all(vals[i:]) for i, correct in enumerate(vals[:-1]))


def question_type(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("what color") or " color " in q or q.endswith(" color?"):
        return "color"
    if q.startswith("how many") or q.startswith("number of"):
        return "count"
    if q.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ")):
        return "yes_no"
    if q.startswith("where"):
        return "where"
    if q.startswith("who"):
        return "who"
    if q.startswith("which"):
        return "which"
    if q.startswith("what"):
        return "what"
    return "other"


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def bootstrap_ci(values: Iterable[float], iters: int = 1000, seed: int = 0):
    vals = np.asarray(list(values), dtype=float)
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(iters)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def budget_distribution(budgets: Iterable[float]) -> Dict[str, int]:
    return {ratio_key(k): v for k, v in sorted(Counter(float(b) for b in budgets).items())}


def evaluate_budget_assignment(records: List[Dict], budgets: List[float]) -> Dict:
    if len(records) != len(budgets):
        raise ValueError("records and budgets must have equal length")
    correct = []
    normalized_budgets = []
    for record, budget in zip(records, budgets):
        key = ratio_key(budget)
        if key not in record["correct"]:
            raise KeyError(f"budget {key} missing from record")
        correct.append(float(record["correct"][key]))
        normalized_budgets.append(float(budget))
    return {
        "acc": mean(correct),
        "avg_budget": mean(normalized_budgets),
        "budget_distribution": budget_distribution(normalized_budgets),
    }


def fixed_budget(records: List[Dict], budget: float) -> Dict:
    return evaluate_budget_assignment(records, [budget] * len(records))
