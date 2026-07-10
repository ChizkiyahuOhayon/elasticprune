# TextVQA External-Validity Run on 8×3090

Purpose: test whether the failed GQA response predictor is benchmark-specific. This run collects the complete 5000-question TextVQA validation response matrix; it does not run the failed TokenMarket router.

## 1. Update and activate

```bash
cd elasticprune
git pull
conda activate elastic
export HF_ENDPOINT=https://hf-mirror.com
```

## 2. Required smoke run

```bash
bash scripts/run_textvqa_external_oracle.sh smoke
```

Do not start the full run unless this produces:

```text
capture reuse smoke test passed
TextVQA oracle smoke test passed
[textvqa] done: textvqa_external_smoke.tar.gz
```

Also inspect:

```bash
tail -n 50 results_textvqa_external_smoke/logs_oracle_0.log
```

## 3. Full eight-GPU run

```bash
nohup bash scripts/run_textvqa_external_oracle.sh full \
  > textvqa_external_full.log 2>&1 &
```

Monitor:

```bash
tail -f textvqa_external_full.log
tail -f results_textvqa_external_full/logs_oracle_0.log
```

The full run uses all 5000 validation questions and shards them across GPUs 0–7. Every sample performs one selector capture, six pruned generations, and one true-full generation.

If a shard fails, rerun the same `full` command. Atomic partial checkpoints and `--resume` prevent completed samples from being repeated.

## 4. Return artifact

When complete, send back:

```text
textvqa_external_full.tar.gz
```

The archive includes raw budget predictions, official TextVQA soft scores, full predictions, image IDs, selector boundary margins, and cross-layer attention-stability signals.

## 5. Optional GPU selection

Override GPU IDs if needed:

```bash
GPU_IDS=0,2,3,4,5,6,7,8 bash scripts/run_textvqa_external_oracle.sh full
```
