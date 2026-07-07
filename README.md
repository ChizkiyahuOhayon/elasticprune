# ElasticPrune 代码脚手架

## 结构

```
code/
├── setup_env.sh              # 环境搭建（conda + lmms-eval + baseline 仓库）
├── elasticprune/
│   ├── pruning.py            # FastV 风格剪枝，支持每样本 keep_ratio（两遍实现）
│   ├── signals.py            # 预算信号：视觉冗余度 + 查询特异性
│   ├── budget.py             # 难度分 → 保留率的分位数映射（保证平均预算守恒）
│   └── smoke_test.py         # 单卡全链路冒烟测试
└── scripts/
    ├── oracle_gqa.py         # ★ Go/No-Go 实验：oracle 预算上界
    └── run_eval_matrix.sh    # 8 卡并行 benchmark 矩阵（lmms-eval）
```

## 本周执行顺序

1. `bash setup_env.sh`
2. `CUDA_VISIBLE_DEVICES=0 python -m elasticprune.smoke_test` — 验证剪枝生成能跑、输出合理
3. 8 卡跑 oracle（每卡一个 shard）：
   ```bash
   for i in 0 1 2 3 4 5 6 7; do
     CUDA_VISIBLE_DEVICES=$i python scripts/oracle_gqa.py \
       --n 2000 --shard $i --num-shards 8 --out results/oracle.json &
   done; wait
   ```
4. 看输出中 `oracle: acc X @ avg budget Y` vs 各固定比例的 acc——**同等平均预算下 oracle 增益 ≥2% 才继续**
5. `bash scripts/run_eval_matrix.sh` 跑无剪枝 baseline 全矩阵，建立参照

## 已知注意事项（未在真机验证，预期需要小修）

- **transformers 版本敏感**：`pruning.py` 假设 input_ids 中 `<image>` 已展开（4.44+ 行为）；
  `position_ids` 传入 `generate` 在部分版本报错 → 先 `keep_positions=False` 跑通再对比
- `attn_implementation="eager"` 是必须的（需要注意力图），速度慢是预期内的——oracle 阶段无所谓
- GQA 数据集字段名以 `lmms-lab/GQA` 实际 schema 为准，`imageId`/`id` 可能需对照调整
- 3090 上 llava-1.5-7b bf16 约占 15GB，单卡余量充足；13B 需要注意 eager 注意力的激活显存
- lmms-eval 的 task 名（`mmbench_en_dev` 等）以 `python -m lmms_eval --tasks list` 输出为准

## Oracle 结果判读

| 情形 | 行动 |
|---|---|
| oracle 增益 ≥5% | 全速推进，信号预测器只需吃到一半即成文 |
| 2-5% | 继续，但把"悬崖分析+oracle"提为论文主贡献 |
| <2% | 止损，切换 idea B（无注意力+vLLM 部署评测）|
