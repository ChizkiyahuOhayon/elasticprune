"""Budget routers for offline and online ElasticPrune experiments.

Routers in this module assign one of the scanned token budgets to each sample.
They intentionally do not know the ground-truth answer. Offline analysis can
evaluate these assignments against saved oracle records, while online scripts
can reuse the same routing rules before generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np

from elasticprune.eval_utils import normalize_text, ratio_key

SCANNED_BUDGETS: List[float] = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]


class BudgetRouter:
    """Base class for per-sample budget routers."""

    name: str = "router"

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        raise NotImplementedError

    def cumulative_costs(self, records: List[Dict], budgets: List[float]) -> Optional[List[float]]:
        """Return per-sample cumulative probe cost if the router is multi-pass.

        Single-pass routers return None, in which case final budget is also the
        only budget cost considered by the offline evaluator.
        """
        return None


@dataclass(frozen=True)
class FixedBudgetRouter(BudgetRouter):
    budget: float

    @property
    def name(self) -> str:
        return f"fixed_{ratio_key(self.budget)}"

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        return [self.budget] * len(records)


@dataclass(frozen=True)
class RandomAdaptiveRouter(BudgetRouter):
    target_budget: float
    seed: int = 0
    min_budget: float = 0.02

    @property
    def name(self) -> str:
        return f"random_adaptive_{ratio_key(self.target_budget)}_seed{self.seed}"

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        rng = np.random.default_rng(self.seed)
        budgets = np.full(len(records), self.min_budget)
        order = np.arange(len(records))
        rng.shuffle(order)
        higher_levels = [b for b in SCANNED_BUDGETS if b > self.min_budget]
        p = 0
        while budgets.mean() < self.target_budget and p < len(order):
            idx = order[p]
            higher = [b for b in higher_levels if b > budgets[idx]]
            if not higher:
                p += 1
                continue
            previous = budgets[idx]
            budgets[idx] = higher[0]
            if budgets.mean() > self.target_budget:
                budgets[idx] = previous
                p += 1
        return budgets.tolist()


@dataclass(frozen=True)
class QuantileFeatureRouter(BudgetRouter):
    feature: str
    target_budget: float
    higher_is_harder: bool = True
    min_budget: float = 0.02

    @property
    def name(self) -> str:
        direction = "high_hard" if self.higher_is_harder else "low_hard"
        return f"{self.feature}_{direction}_{ratio_key(self.target_budget)}"

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        if rows is None:
            raise ValueError("QuantileFeatureRouter requires labeled rows")
        values = np.asarray([row[self.feature] for row in rows], dtype=float)
        if not self.higher_is_harder:
            values = -values
        order = np.argsort(values)
        budgets = np.full(len(rows), self.min_budget)
        higher_levels = [b for b in SCANNED_BUDGETS if b > self.min_budget]
        idx = len(order) - 1
        while budgets.mean() < self.target_budget and idx >= 0:
            sample_idx = order[idx]
            higher = [b for b in higher_levels if b > budgets[sample_idx]]
            if not higher:
                idx -= 1
                continue
            previous = budgets[sample_idx]
            budgets[sample_idx] = higher[0]
            if budgets.mean() > self.target_budget:
                budgets[sample_idx] = previous
                idx -= 1
        return budgets.tolist()


@dataclass(frozen=True)
class QuestionTypeRouter(BudgetRouter):
    mapping: Mapping[str, float]
    default_budget: float = 0.25
    router_name: str = "question_type"

    @property
    def name(self) -> str:
        return self.router_name

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        if rows is None:
            raise ValueError("QuestionTypeRouter requires labeled rows")
        return [
            float(self.mapping.get(row["question_type"], self.default_budget))
            for row in rows
        ]


@dataclass(frozen=True)
class PairAgreementRouter(BudgetRouter):
    low_budget: float
    high_budget: float
    fallback_budget: float

    @property
    def name(self) -> str:
        return (
            f"pair_{ratio_key(self.low_budget)}_{ratio_key(self.high_budget)}"
            f"_else_{ratio_key(self.fallback_budget)}"
        )

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        budgets = []
        low_key = ratio_key(self.low_budget)
        high_key = ratio_key(self.high_budget)
        for record in records:
            low_pred = normalize_text(record["pred"][low_key])
            high_pred = normalize_text(record["pred"][high_key])
            budgets.append(self.low_budget if low_pred == high_pred else self.fallback_budget)
        return budgets

    def cumulative_costs(self, records: List[Dict], budgets: List[float]) -> List[float]:
        return [
            self.low_budget + self.high_budget
            + (self.fallback_budget if budget == self.fallback_budget else 0.0)
            for budget in budgets
        ]


@dataclass(frozen=True)
class AgreementFallbackRouter(BudgetRouter):
    """Probe agreement router with query-type-specific fallback budgets.

    The router first compares predictions at low_budget and high_budget. If the
    normalized predictions agree, it accepts low_budget. Otherwise, it falls back
    to a budget selected by question_type.
    """

    low_budget: float
    high_budget: float
    fallback_by_type: Mapping[str, float]
    default_fallback: float = 0.25
    router_name: str = "agreement_fallback"

    @property
    def name(self) -> str:
        return self.router_name

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        if rows is None:
            raise ValueError("AgreementFallbackRouter requires labeled rows")
        budgets = []
        low_key = ratio_key(self.low_budget)
        high_key = ratio_key(self.high_budget)
        for record, row in zip(records, rows):
            low_pred = normalize_text(record["pred"][low_key])
            high_pred = normalize_text(record["pred"][high_key])
            if low_pred == high_pred:
                budgets.append(self.low_budget)
            else:
                budgets.append(float(self.fallback_by_type.get(row["question_type"], self.default_fallback)))
        return budgets

    def cumulative_costs(self, records: List[Dict], budgets: List[float]) -> List[float]:
        costs = []
        already_probed = {self.low_budget, self.high_budget}
        for budget in budgets:
            cost = self.low_budget + self.high_budget
            if budget not in already_probed:
                cost += budget
            costs.append(cost)
        return costs


@dataclass(frozen=True)
class CascadeAgreementRouter(BudgetRouter):
    stages: List[float]

    @property
    def name(self) -> str:
        return "cascade_" + "_".join(ratio_key(stage) for stage in self.stages)

    def route(self, records: List[Dict], rows: Optional[List[Dict]] = None) -> List[float]:
        final_budgets = []
        for record in records:
            previous = None
            final_budget = self.stages[-1]
            for stage in self.stages:
                pred = normalize_text(record["pred"][ratio_key(stage)])
                final_budget = stage
                if previous is not None and pred == previous:
                    break
                previous = pred
            final_budgets.append(final_budget)
        return final_budgets

    def cumulative_costs(self, records: List[Dict], budgets: List[float]) -> List[float]:
        costs = []
        for budget in budgets:
            total = 0.0
            for stage in self.stages:
                total += stage
                if stage == budget:
                    break
            costs.append(total)
        return costs
