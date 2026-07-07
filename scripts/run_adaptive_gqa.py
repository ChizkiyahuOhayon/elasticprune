"""主实验：自适应预算 vs 固定预算（oracle 通过 go/no-go 后跑）。

流程：前 calib_n 个样本算难度分校准 BudgetMapper，其余样本上对比
  fixed(target_ratio) vs adaptive(同平均预算)。
用法（8 卡切片，同 oracle）:
  CUDA_VISIBLE_DEVICES=$i python scripts/run_adaptive_gqa.py \
    --n 2000 --target 0.1 --shard $i --num-shards 8
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor
from elasticprune.pruning import prune_generate
from elasticprune.signals import signal_redundancy, signal_specificity, difficulty_score
from elasticprune.budget import BudgetMapper

MODEL = "llava-hf/llava-1.5-7b-hf"


def exact_match(pred, answer):
    return answer.strip().lower() in pred.strip().lower()


@torch.no_grad()
def sample_difficulty(model, inputs):
    img_feats = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )[0]
    txt_ids = inputs["input_ids"][0]
    txt_ids = txt_ids[txt_ids != model.config.image_token_index]
    q_embeds = model.get_input_embeddings()(txt_ids)
    return difficulty_score(signal_redundancy(img_feats),
                            signal_specificity(img_feats, q_embeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="results/adaptive.json")
    args = ap.parse_args()

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(MODEL)
    model.eval()

    ds = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    img_ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    id2img = {r["id"]: i for i, r in enumerate(img_ds)}

    def prep(ex):
        image = img_ds[id2img[ex["imageId"]]]["image"]
        prompt = f"USER: <image>\n{ex['question']}\nAnswer the question using a single word or phrase. ASSISTANT:"
        return processor(images=image, text=prompt, return_tensors="pt").to("cuda")

    # 校准（所有 shard 用同一段前缀样本，保证映射一致）
    calib = ds.select(range(args.calib_n))
    scores = [sample_difficulty(model, prep(ex)) for ex in calib]
    mapper = BudgetMapper().fit(scores, target_ratio=args.target)

    # 评测段
    eval_ds = ds.select(range(args.calib_n, min(args.calib_n + args.n, len(ds))))
    eval_ds = eval_ds.shard(num_shards=args.num_shards, index=args.shard)

    records = []
    for ex in eval_ds:
        inputs = prep(ex)
        r_adapt = mapper.ratio(sample_difficulty(model, inputs))
        rec = {"qid": ex["id"], "ratio_adaptive": r_adapt}
        for name, r in [("fixed", args.target), ("adaptive", r_adapt)]:
            out = prune_generate(model, inputs, keep_ratio=r, max_new_tokens=16)
            pred = processor.decode(out[0], skip_special_tokens=True)
            rec[name] = exact_match(pred, ex["answer"])
        records.append(rec)
        if len(records) % 25 == 0:
            print(f"[shard {args.shard}] {len(records)} done")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out.replace(".json", f".shard{args.shard}.json"), "w") as f:
        json.dump(records, f)

    n = len(records)
    print(f"fixed({args.target}): {sum(x['fixed'] for x in records)/n:.4f}")
    print(f"adaptive(avg {sum(x['ratio_adaptive'] for x in records)/n:.3f}): "
          f"{sum(x['adaptive'] for x in records)/n:.4f}")


if __name__ == "__main__":
    main()
