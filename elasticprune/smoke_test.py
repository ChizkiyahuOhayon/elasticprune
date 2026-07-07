"""冒烟测试：单卡验证 剪枝生成 + 信号计算 全链路能跑。
用法: CUDA_VISIBLE_DEVICES=0 python -m elasticprune.smoke_test"""
import requests, torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from elasticprune.pruning import prune_generate, full_generate
from elasticprune.signals import signal_redundancy, signal_specificity, difficulty_score

MODEL = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL)
model.eval()

image = Image.open(requests.get(
    "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
prompt = "USER: <image>\nWhat animals are in this image? ASSISTANT:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

print("== full ==")
out = full_generate(model, inputs)
print(processor.decode(out[0], skip_special_tokens=True))

for r in [0.5, 0.1, 0.05]:
    out = prune_generate(model, inputs, keep_ratio=r)
    print(f"== keep {r} ==\n{processor.decode(out[0], skip_special_tokens=True)}")

# 信号
with torch.no_grad():
    img_feats = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )[0]  # (576, D)
    txt_ids = inputs["input_ids"][0]
    txt_ids = txt_ids[txt_ids != model.config.image_token_index]
    q_embeds = model.get_input_embeddings()(txt_ids)

red = signal_redundancy(img_feats)
spec = signal_specificity(img_feats, q_embeds)
print(f"redundancy={red:.3f} specificity={spec:.3f} "
      f"difficulty={difficulty_score(red, spec):.3f}")
print("冒烟测试通过 ✔")
