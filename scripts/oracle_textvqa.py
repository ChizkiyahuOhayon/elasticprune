"""Collect a TextVQA visual-token budget-response matrix.

The selector attention is captured once per sample and reused for all budgets.
This preserves the existing two-pass FastV-style token choice while avoiding a
redundant selector forward for every keep ratio.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

from elasticprune.pruning import (
    _capture_merged_embeds,
    full_generate,
    prune_generate_from_capture,
)
from elasticprune.signals import (
    signal_budget_boundaries,
    signal_distribution_concentration,
    signal_pair_stability,
    signal_redundancy,
    signal_specificity,
)


RATIOS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
MODEL = "llava-hf/llava-1.5-7b-hf"
DATASET = "lmms-lab/textvqa"
SPLIT = "validation"


def atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary, path)


def select_indices(length, n, seed):
    n = min(int(n), int(length))
    if n == length:
        return list(range(length))
    return random.Random(seed).sample(range(length), n)


def textvqa_soft_score(prediction, answers, answer_processor):
    prediction = answer_processor(prediction)
    references = [answer_processor(answer) for answer in answers]
    accuracies = []
    for index in range(len(references)):
        other_answers = references[:index] + references[index + 1 :]
        matches = sum(answer == prediction for answer in other_answers)
        accuracies.append(min(1.0, matches / 3.0))
    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def decode_new_tokens(processor, output_ids):
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def compute_signals_from_capture(model, inputs, merged_embeds, attentions,
                                 prune_layer=2):
    input_ids = inputs["input_ids"]
    image_mask = input_ids[0] == model.config.image_token_index
    if not image_mask.any():
        return {}

    image_features = merged_embeds[0, image_mask]
    query_features = merged_embeds[0, ~image_mask]
    result = {
        "redundancy_erank": signal_redundancy(image_features),
        "query_specificity_entropy": signal_specificity(image_features, query_features),
        "n_image_tokens": int(image_mask.sum().item()),
        "n_text_tokens": int((~image_mask).sum().item()),
    }

    layer_scores = {}
    for layer in range(min(5, len(attentions))):
        scores = attentions[layer][0, :, -1, :].mean(0)[image_mask]
        layer_scores[layer] = scores
        result.update(
            signal_distribution_concentration(scores, prefix=f"attn_l{layer}_")
        )

    selector_scores = layer_scores.get(prune_layer)
    if selector_scores is not None:
        result.update(
            signal_budget_boundaries(selector_scores, RATIOS, prefix="selector_")
        )
    for first, second in ((1, 2), (2, 3)):
        if first in layer_scores and second in layer_scores:
            result.update(
                signal_pair_stability(
                    layer_scores[first],
                    layer_scores[second],
                    keep_ratio=0.25,
                    prefix=f"attn_l{first}_l{second}_",
                )
            )
    return result


def summarize(records):
    if not records:
        return
    print(f"records: {len(records)}")
    print(f"images: {len(set(record['imageId'] for record in records))}")
    for ratio in RATIOS:
        key = str(ratio)
        score = sum(record["score"][key] for record in records) / len(records)
        positive = sum(record["correct"][key] for record in records) / len(records)
        print(f"fixed {ratio:>4}: soft score {score:.4f}  positive {positive:.4f}")
    full_score = sum(record["full_score"] for record in records) / len(records)
    full_positive = sum(record["full_correct"] for record in records) / len(records)
    print(f"full: soft score {full_score:.4f}  positive {full_positive:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out", default="results_textvqa_external/oracle.json")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("shard index must be in [0, num_shards)")

    from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL)
    answer_processor = EvalAIAnswerProcessor()
    model.eval()

    dataset = load_dataset(DATASET, split=SPLIT)
    indices = select_indices(len(dataset), args.n, args.seed)
    dataset = dataset.select(indices)
    dataset = dataset.shard(num_shards=args.num_shards, index=args.shard)

    partial_path = args.out.replace(".json", f".shard{args.shard}.partial.json")
    final_path = args.out.replace(".json", f".shard{args.shard}.json")
    records = []
    if args.resume:
        resume_path = final_path if os.path.isfile(final_path) else partial_path
        if os.path.isfile(resume_path):
            with open(resume_path, encoding="utf-8") as handle:
                records = json.load(handle)
            print(f"[shard {args.shard}] resumed {len(records)} records from {resume_path}")
    completed = {str(record["qid"]) for record in records}

    for example in dataset:
        qid = str(example["question_id"])
        if qid in completed:
            continue
        try:
            image = example["image"].convert("RGB")
            question = str(example["question"]).capitalize()
            prompt = (
                f"USER: <image>\n{question}\n"
                "Answer the question using a single word or phrase. ASSISTANT:"
            )
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
            merged_embeds, attentions = _capture_merged_embeds(model, inputs)
            record = {
                "qid": qid,
                "imageId": str(example["image_id"]),
                "question": example["question"],
                "answers": list(example["answers"]),
                "pred": {},
                "score": {},
                "correct": {},
                "signals": compute_signals_from_capture(
                    model, inputs, merged_embeds, attentions
                ),
                "image_width": int(example.get("image_width", image.width)),
                "image_height": int(example.get("image_height", image.height)),
            }

            for ratio in RATIOS:
                output = prune_generate_from_capture(
                    model,
                    inputs,
                    merged_embeds,
                    attentions,
                    keep_ratio=ratio,
                    max_new_tokens=args.max_new_tokens,
                )
                prediction = decode_new_tokens(processor, output)
                score = textvqa_soft_score(
                    prediction, example["answers"], answer_processor
                )
                record["pred"][str(ratio)] = prediction
                record["score"][str(ratio)] = score
                record["correct"][str(ratio)] = score > 0

            full_output = full_generate(
                model, inputs, max_new_tokens=args.max_new_tokens
            )
            full_prediction = decode_new_tokens(processor, full_output)
            full_score = textvqa_soft_score(
                full_prediction, example["answers"], answer_processor
            )
            record["full_pred"] = full_prediction
            record["full_score"] = full_score
            record["full_correct"] = full_score > 0
            record["prune1_matches_full"] = (
                record["pred"]["1.0"] == full_prediction
            )
            records.append(record)
            completed.add(qid)
        except Exception:
            atomic_json_dump(records, partial_path)
            print(
                f"[shard {args.shard}] failed qid={qid}; checkpoint saved to {partial_path}",
                file=sys.stderr,
            )
            raise

        if len(records) % args.save_every == 0:
            atomic_json_dump(records, partial_path)
            print(f"[shard {args.shard}] {len(records)} done", flush=True)

    atomic_json_dump(records, final_path)
    summarize(records)


if __name__ == "__main__":
    main()
