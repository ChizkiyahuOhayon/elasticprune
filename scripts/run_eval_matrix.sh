#!/bin/bash
# 8 卡并行评测矩阵：每卡一个 benchmark（lmms-eval）
# 用法: bash scripts/run_eval_matrix.sh llava-hf/llava-1.5-7b-hf
set -e
MODEL=${1:-llava-hf/llava-1.5-7b-hf}
TASKS=(gqa mmbench_en_dev mme pope textvqa_val scienceqa_img seedbench mmvet)

mkdir -p logs results
for i in "${!TASKS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python -m lmms_eval \
    --model llava_hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks "${TASKS[$i]}" \
    --batch_size 1 \
    --output_path "results/${TASKS[$i]}" \
    > "logs/${TASKS[$i]}.log" 2>&1 &
done
wait
echo "全部完成，结果在 results/"
