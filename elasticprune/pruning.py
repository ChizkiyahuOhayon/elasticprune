"""FastV 风格的视觉 token 剪枝（支持每样本自适应保留率）。

两遍实现（oracle/研究阶段用，正确性优先，不追求速度）：
  Pass 1: 完整前向，拿到 prune_layer 处最后一个 prompt token 对图像 token 的注意力
  Pass 2: 按 keep_ratio 保留 top-k 图像 token，用 inputs_embeds 重新生成

适用 llava-hf 系列 (llava-1.5-7b-hf 等, transformers>=4.44,
input_ids 中 <image> 已展开为 576 个 image_token_index)。
"""
import torch


def _capture_merged_embeds(model, inputs):
    """Hook 第一个 decoder layer 的输入，拿到视觉特征合并后的 embeds。"""
    captured = {}

    def hook(module, args, kwargs):
        captured["embeds"] = kwargs.get("hidden_states", args[0] if args else None)
        return None

    layer0 = model.language_model.model.layers[0]
    h = layer0.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        out = model(**inputs, output_attentions=True, use_cache=False)
    finally:
        h.remove()
    return captured["embeds"], out.attentions


@torch.no_grad()
def prune_generate(model, inputs, keep_ratio: float, prune_layer: int = 2,
                   max_new_tokens: int = 64, keep_positions: bool = True):
    """按 keep_ratio 剪枝图像 token 后生成。keep_ratio=1.0 等价于不剪。

    返回 generated_ids（仅新生成部分）。
    """
    input_ids = inputs["input_ids"]  # (1, L)
    assert input_ids.shape[0] == 1, "研究阶段先只支持 batch=1"
    img_mask = input_ids[0] == model.config.image_token_index  # (L,)
    img_pos = img_mask.nonzero(as_tuple=True)[0]
    n_img = len(img_pos)
    n_keep = max(1, int(n_img * keep_ratio))

    # Pass 1
    merged_embeds, attentions = _capture_merged_embeds(model, inputs)
    # 最后一个 token 对所有位置的注意力, head 均值: (L,)
    attn = attentions[prune_layer][0, :, -1, :].mean(0)
    img_scores = attn[img_pos]
    topk = img_scores.topk(n_keep).indices
    kept_img_pos = img_pos[topk.sort().values]

    # 组装保留位置：全部文本 + 保留的图像 token，按原顺序
    all_pos = torch.arange(input_ids.shape[1], device=input_ids.device)
    text_pos = all_pos[~img_mask]
    kept = torch.cat([text_pos, kept_img_pos]).sort().values

    pruned_embeds = merged_embeds[:, kept, :]
    attn_mask = torch.ones(1, len(kept), dtype=torch.long, device=input_ids.device)

    gen_kwargs = dict(inputs_embeds=pruned_embeds, attention_mask=attn_mask,
                      max_new_tokens=max_new_tokens, do_sample=False)
    if keep_positions:
        # 保留原始 RoPE 位置（与 FastV 一致）。cache_position/position_ids 支持
        # 因 transformers 版本而异，若报错先置 keep_positions=False 对比
        gen_kwargs["position_ids"] = kept.unsqueeze(0)

    out = model.language_model.generate(**gen_kwargs)
    return out


@torch.no_grad()
def full_generate(model, inputs, max_new_tokens: int = 64):
    """不剪枝的对照。"""
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return out[:, inputs["input_ids"].shape[1]:]
