"""Go/No-Go 实验：GQA 子集上的 oracle 预算上界。

对每个样本扫描 keep_ratio ∈ {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}，
oracle = 每样本取"能答对的最小预算"；对比同平均预算下的固定比例。

判据：若 oracle 在同平均预算下相对固定比例增益 < 2%，止损换 idea B。

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
from elasticprune.pruning import prune_generate

RATIOS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
MODEL = "llava-hf/llava-1.5-7b-hf"


def exact_match(pred: str, answer: str) -> bool:
    return answer.strip().lower() in pred.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="results/oracle.json")
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

        rec = {"qid": ex["id"], "answer": ex["answer"], "correct": {}}
        for r in RATIOS:
            out = prune_generate(model, inputs, keep_ratio=r, max_new_tokens=16)
            pred = processor.decode(out[0], skip_special_tokens=True)
            rec["correct"][str(r)] = exact_match(pred, ex["answer"])
        records.append(rec)
        if len(records) % 25 == 0:
            print(f"[shard {args.shard}] {len(records)} done")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out.replace(".json", f".shard{args.shard}.json"), "w") as f:
        json.dump(records, f)

    # 汇总（单 shard 内）
    for r in RATIOS:
        acc = sum(x["correct"][str(r)] for x in records) / len(records)
        print(f"fixed ratio {r:>5}: acc {acc:.3f}")
    # oracle: 每样本最小可答对预算；答不对的记 1.0
    budgets, correct = [], 0
    for x in records:
        ok = [r for r in RATIOS if x["correct"][str(r)]]
        budgets.append(min(ok) if ok else 1.0)
        correct += bool(ok)
    print(f"oracle: acc {correct/len(records):.3f} @ avg budget {sum(budgets)/len(budgets):.3f}")


if __name__ == "__main__":
    main()
