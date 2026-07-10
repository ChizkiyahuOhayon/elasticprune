"""Summarize TextVQA budget-response shards without model dependencies."""

import argparse
import glob
import json
import os
import string
from collections import Counter


RATIOS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text):
    return " ".join(str(text).strip().lower().translate(PUNCTUATION).split())


def load_records(result_dir):
    files = sorted(
        path
        for path in glob.glob(os.path.join(result_dir, "oracle.shard*.json"))
        if ".partial." not in path
    )
    records = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            records.extend(json.load(handle))
    return records, files


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results_textvqa_external")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    records, files = load_records(args.dir)
    if not records:
        raise SystemExit("no final TextVQA oracle shards found")
    qids = [str(record["qid"]) for record in records]
    if len(qids) != len(set(qids)):
        raise SystemExit("duplicate qids detected")

    fixed_soft = {
        str(ratio): mean(record["score"][str(ratio)] for record in records)
        for ratio in RATIOS
    }
    fixed_positive = {
        str(ratio): mean(record["score"][str(ratio)] > 0 for record in records)
        for ratio in RATIOS
    }
    behavior = {
        str(ratio): mean(
            normalize(record["pred"][str(ratio)]) == normalize(record["full_pred"])
            for record in records
        )
        for ratio in RATIOS
    }
    task_oracle_scores = [
        max(record["score"][str(ratio)] for ratio in RATIOS) for record in records
    ]
    task_oracle_budgets = []
    nonmonotonic = []
    patterns = Counter()
    for record in records:
        scores = [float(record["score"][str(ratio)]) for ratio in RATIOS]
        best = max(scores)
        task_oracle_budgets.append(
            next((ratio for ratio, score in zip(RATIOS, scores) if score == best), 1.0)
            if best > 0
            else 1.0
        )
        nonmonotonic.append(
            any(score > later + 1e-12 for index, score in enumerate(scores[:-1])
                for later in scores[index + 1 :])
        )
        patterns[tuple(score > 0 for score in scores)] += 1

    summary = {
        "n_records": len(records),
        "n_images": len(set(record["imageId"] for record in records)),
        "n_shards": len(files),
        "fixed_soft_accuracy": fixed_soft,
        "fixed_positive_rate": fixed_positive,
        "full_soft_accuracy": mean(record["full_score"] for record in records),
        "full_positive_rate": mean(record["full_score"] > 0 for record in records),
        "behavior_preservation": behavior,
        "task_oracle_soft_accuracy": mean(task_oracle_scores),
        "task_oracle_avg_budget": mean(task_oracle_budgets),
        "nonmonotonic_rate": mean(nonmonotonic),
        "prune1_full_exact_match": mean(
            record.get("prune1_matches_full", False) for record in records
        ),
        "top_positive_patterns": [
            {"pattern": "".join("1" if value else "0" for value in pattern), "count": count}
            for pattern, count in patterns.most_common(10)
        ],
        "source_files": files,
    }
    output = args.out or os.path.join(args.dir, "textvqa_oracle_summary.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"records: {summary['n_records']}  images: {summary['n_images']}")
    for ratio in RATIOS:
        key = str(ratio)
        print(
            f"fixed {ratio:>4}: soft {fixed_soft[key]:.4f}  "
            f"positive {fixed_positive[key]:.4f}  behavior {behavior[key]:.4f}"
        )
    print(
        f"full soft {summary['full_soft_accuracy']:.4f}  "
        f"task oracle {summary['task_oracle_soft_accuracy']:.4f} "
        f"@ {summary['task_oracle_avg_budget']:.4f}"
    )
    print(f"non-monotonic: {summary['nonmonotonic_rate']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
