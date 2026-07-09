"""Smoke tests for offline budget router utilities.

Run:
  python -m elasticprune.router_smoke_test
"""

from __future__ import annotations

from elasticprune.eval_utils import evaluate_budget_assignment
from elasticprune.routers import (
    AgreementFallbackRouter,
    CascadeAgreementRouter,
    FixedBudgetRouter,
    PairAgreementRouter,
    QuantileFeatureRouter,
    QuestionTypeRouter,
    RandomAdaptiveRouter,
)


def make_records():
    return [
        {
            "answer": "yes",
            "question": "Is it sunny?",
            "correct": {"0.02": True, "0.05": True, "0.1": True, "0.25": True, "0.5": True, "1.0": True},
            "pred": {"0.02": "yes", "0.05": "yes", "0.1": "yes", "0.25": "yes", "0.5": "yes", "1.0": "yes"},
        },
        {
            "answer": "red",
            "question": "What color is the car?",
            "correct": {"0.02": False, "0.05": False, "0.1": True, "0.25": True, "0.5": True, "1.0": True},
            "pred": {"0.02": "blue", "0.05": "green", "0.1": "red", "0.25": "red", "0.5": "red", "1.0": "red"},
        },
        {
            "answer": "two",
            "question": "How many dogs are there?",
            "correct": {"0.02": False, "0.05": False, "0.1": False, "0.25": True, "0.5": True, "1.0": True},
            "pred": {"0.02": "one", "0.05": "one", "0.1": "three", "0.25": "two", "0.5": "two", "1.0": "two"},
        },
    ]


def make_rows():
    return [
        {"question_type": "yes_no", "score": 0.1},
        {"question_type": "color", "score": 0.5},
        {"question_type": "count", "score": 0.9},
    ]


def main():
    records = make_records()
    rows = make_rows()

    fixed = FixedBudgetRouter(0.10).route(records, rows)
    assert fixed == [0.10, 0.10, 0.10]
    assert evaluate_budget_assignment(records, fixed)["acc"] == 2 / 3

    question = QuestionTypeRouter({"yes_no": 0.02, "color": 0.10}, default_budget=0.25)
    assert question.route(records, rows) == [0.02, 0.10, 0.25]
    assert evaluate_budget_assignment(records, question.route(records, rows))["acc"] == 1.0

    pair = PairAgreementRouter(0.02, 0.05, 0.25)
    pair_budgets = pair.route(records, rows)
    assert pair_budgets == [0.02, 0.25, 0.02]
    assert pair.cumulative_costs(records, pair_budgets) == [0.07, 0.32, 0.07]

    agreement_fallback = AgreementFallbackRouter(
        0.02,
        0.05,
        {"color": 0.10, "count": 0.25},
        default_fallback=0.50,
    )
    fallback_budgets = agreement_fallback.route(records, rows)
    assert fallback_budgets == [0.02, 0.10, 0.02]
    assert agreement_fallback.cumulative_costs(records, fallback_budgets) == [0.07, 0.17, 0.07]

    cascade = CascadeAgreementRouter([0.02, 0.10, 0.25])
    assert cascade.route(records, rows) == [0.10, 0.25, 0.25]

    quantile = QuantileFeatureRouter("score", target_budget=0.10, higher_is_harder=True)
    quantile_budgets = quantile.route(records, rows)
    assert len(quantile_budgets) == len(records)
    assert sum(quantile_budgets) / len(quantile_budgets) <= 0.10

    random_router = RandomAdaptiveRouter(target_budget=0.10, seed=0)
    random_budgets = random_router.route(records, rows)
    assert len(random_budgets) == len(records)
    assert all(b in {0.02, 0.05, 0.10, 0.25, 0.50, 1.00} for b in random_budgets)

    print("router smoke test passed")


if __name__ == "__main__":
    main()
