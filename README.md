# ElasticPrune 代码脚手架

## 结构

```
code/
├── setup_env.sh              # 环境搭建（conda + lmms-eval + baseline 仓库）
├── elasticprune/
│   ├── pruning.py            # FastV 风格剪枝，支持每样本 keep_ratio（两遍实现）
│   ├── signals.py            # 预算信号：视觉冗余度 + 查询特异性
│   ├── budget.py             # 难度分 → 保留率的分位数映射（保证平均预算守恒）
│   ├── eval_utils.py         # oracle 记录的离线评估工具
│   ├── routers.py            # fixed/random/heuristic/stability budget routers
│   ├── router_smoke_test.py  # 不依赖 GPU 的 router 单元级冒烟测试
│   └── smoke_test.py         # 单卡全链路冒烟测试
└── scripts/
    ├── oracle_gqa.py         # ★ Go/No-Go 实验：oracle 预算上界
    ├── analyze_oracle.py     # oracle 结果分析：task/preservation/correction/non-monotonic
    ├── analyze_router_offline.py # 离线路由器可行性分析（不跑模型）
    ├── search_router_offline.py  # train/test 离线搜索，选择是否进入真实 GPU router 实验
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
5. 用 oracle v2 结果先做离线路由器诊断，不要直接把 naive adaptive router 发到 8 卡：
   ```bash
   python scripts/analyze_router_offline.py \
     --dir results \
     --out results/router_offline.json \
     --csv results/router_labeled_samples.csv \
     --md results/router_offline.md
   ```
6. 用 train/test 搜索检查 router 是否泛化：
   ```bash
   python scripts/search_router_offline.py \
     --dir results \
     --out results/router_search.json \
     --md results/router_search.md
   ```
7. 只有当 held-out split 上的 router 明确打过 fixed/random adaptive baseline 后，再跑真实 router 推理实验。
8. 修改 router 代码后先跑不依赖 GPU 的检查：
   ```bash
   python -m elasticprune.router_smoke_test
   ```
9. `bash scripts/run_eval_matrix.sh` 跑无剪枝 baseline 全矩阵，建立参照

## 已知注意事项（未在真机验证，预期需要小修）

- **transformers 版本敏感**：`pruning.py` 假设 input_ids 中 `<image>` 已展开（4.44+ 行为）；
  `position_ids` 传入 `generate` 在部分版本报错 → 先 `keep_positions=False` 跑通再对比
- `attn_implementation="eager"` 是必须的（需要注意力图），速度慢是预期内的——oracle 阶段无所谓
- GQA 数据集字段名以 `lmms-lab/GQA` 实际 schema 为准，`imageId`/`id` 可能需对照调整
- 3090 上 llava-1.5-7b bf16 约占 15GB，单卡余量充足；13B 需要注意 eager 注意力的激活显存
- lmms-eval 的 task 名（`mmbench_en_dev` 等）以 `python -m lmms_eval --tasks list` 输出为准
- 当前 `redundancy_erank + query_specificity_entropy` 的 naive router 已经在 GQA v2 上离线验证过，预测力不足；第三轮正式 GPU 实验前必须先改进 router signal。

## Oracle 结果判读

| 情形 | 行动 |
|---|---|
| oracle 增益 ≥5% | 全速推进，信号预测器只需吃到一半即成文 |
| 2-5% | 继续，但把"悬崖分析+oracle"提为论文主贡献 |
| <2% | 止损，切换 idea B（无注意力+vLLM 部署评测）|
