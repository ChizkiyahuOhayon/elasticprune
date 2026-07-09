"""GQA budget-response oracle.

对每个样本扫描 keep_ratio ∈ {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}，
保存每个预算的 prediction/correct，并额外保存 true full-generate 结果。

输出可用于拆分三种 oracle:
  1. task oracle: 任意预算答对即可
  2. preservation oracle: full 本来答对时，最小保持正确预算
  3. correction oracle: full 答错但某个剪枝预算答对

用法（单卡）:
  CUDA_VISIBLE_DEVICES=0 python scripts/oracle_gqa.py --n 500 --out results/oracle.json
多卡切片:
  CUDA_VISIBLE_DEVICES=$i python scripts/oracle_gqa.py --n 500 --shard $i --num-shards 8
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor
from elasticprune.pruning import _capture_merged_embeds, full_generate, prune_generate
from elasticprune.signals import signal_redundancy, signal_specificity

RATIOS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
MODEL = "llava-hf/llava-1.5-7b-hf"


def exact_match(pred: str, answer: str) -> bool:
    return answer.strip().lower() in pred.strip().lower()


def decode_new_tokens(processor, output_ids) -> str:
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def compute_free_signals(model, inputs):
    """Compute cheap sample signals from the merged prompt embeddings.

    This uses a single full forward pass. It is for analysis/calibration only;
    the production router should avoid extra forwards unless explicitly measured.
    """
    input_ids = inputs["input_ids"]
    img_mask = input_ids[0] == model.config.image_token_index
    if not img_mask.any():
        return {}

    merged_embeds, _ = _capture_merged_embeds(model, inputs)
    image_features = merged_embeds[0, img_mask]
    query_features = merged_embeds[0, ~img_mask]
    out = {
        "redundancy_erank": signal_redundancy(image_features),
        "query_specificity_entropy": signal_specificity(image_features, query_features),
        "n_image_tokens": int(img_mask.sum().item()),
        "n_text_tokens": int((~img_mask).sum().item()),
    }
    return out


def summarize(records):
    n = len(records)
    if not n:
        return
    for r in RATIOS:
        key = str(r)
        acc = sum(x["correct"][key] for x in records) / n
        print(f"fixed ratio {r:>5}: acc {acc:.3f}")

    full_acc = sum(x.get("full_correct", x["correct"]["1.0"]) for x in records) / n
    print(f"true full_generate: acc {full_acc:.3f}")

    task_budgets, task_correct = [], 0
    preservation_budgets, full_correct_n = [], 0
    correction_n, nonmono_n = 0, 0
    for x in records:
        vals = [bool(x["correct"][str(r)]) for r in RATIOS]
        ok = [r for r, v in zip(RATIOS, vals) if v]
        task_budgets.append(min(ok) if ok else 1.0)
        task_correct += bool(ok)

        full_ok = bool(x.get("full_correct", vals[-1]))
        if full_ok:
            full_correct_n += 1
            preservation_budgets.append(min(ok) if ok else 1.0)
        elif any(vals[:-1]):
            correction_n += 1

        if any(v and not all(vals[i:]) for i, v in enumerate(vals[:-1])):
            nonmono_n += 1

    print(f"task oracle: acc {task_correct/n:.3f} @ avg budget {sum(task_budgets)/n:.3f}")
    if preservation_budgets:
        print(
            "preservation oracle: "
            f"{len(preservation_budgets)}/{full_correct_n} full-correct samples, "
            f"avg min budget {sum(preservation_budgets)/len(preservation_budgets):.3f}"
        )
    print(f"correction samples: {correction_n}/{n} ({correction_n/n:.3f})")
    print(f"non-monotonic samples: {nonmono_n}/{n} ({nonmono_n/n:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="results/oracle.json")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--skip-signals", action="store_true")
    args = ap.parse_args()

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")  # 需要注意力图
    processor = AutoProcessor.from_pretrained(MODEL)
    model.eval()

    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions",
                      split="testdev")
    img_ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images",
                          split="testdev")
    id2img = {r["id"]: i for i, r in enumerate(img_ds)}

    ds = ds.select(range(min(args.n, len(ds))))
    ds = ds.shard(num_shards=args.num_shards, index=args.shard)

    records = []
    for ex in ds:
        image = img_ds[id2img[ex["imageId"]]]["image"]
        prompt = f"USER: <image>\n{ex['question']}\nAnswer the question using a single word or phrase. ASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

        rec = {
            "qid": ex["id"],
            "imageId": ex["imageId"],
            "question": ex["question"],
            "answer": ex["answer"],
            "correct": {},
            "pred": {},
        }
        if not args.skip_signals:
            try:
                rec["signals"] = compute_free_signals(model, inputs)
            except Exception as e:
                rec["signals_error"] = repr(e)

        for r in RATIOS:
            out = prune_generate(model, inputs, keep_ratio=r, max_new_tokens=16)
            pred = decode_new_tokens(processor, out)
            rec["pred"][str(r)] = pred
            rec["correct"][str(r)] = exact_match(pred, ex["answer"])

        try:
            full_out = full_generate(model, inputs, max_new_tokens=16)
            full_pred = decode_new_tokens(processor, full_out)
            rec["full_pred"] = full_pred
            rec["full_correct"] = exact_match(full_pred, ex["answer"])
            rec["prune1_matches_full"] = rec["pred"]["1.0"] == full_pred
        except Exception as e:
            rec["full_error"] = repr(e)

        records.append(rec)
        if len(records) % args.save_every == 0:
            print(f"[shard {args.shard}] {len(records)} done")
            partial = args.out.replace(".json", f".shard{args.shard}.partial.json")
            with open(partial, "w") as f:
                json.dump(records, f, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out.replace(".json", f".shard{args.shard}.json"), "w") as f:
        json.dump(records, f, ensure_ascii=False)

    summarize(records)


if __name__ == "__main__":
    main()
