# 法语 ASR 混合读取（Feature + Wav）说明

本文档说明 `france` recipe 中的混合输入读取能力：同一训练集里同时包含

- 预提取特征 cuts（`input_type=feature`，走 `PrecomputedFeatures`）
- 原始音频 cuts（`input_type=wav`，走 `OnTheFlyFeatures`）

并可与 node producer 采样链路一起使用。

## 1. 开关与入口

训练入口仍是：

- `egs/france/ASR/run_french_150M.sh`

关键参数：

- `--mixed-input-enabled 1`
- `--mixed-input-type-key input_type`
- `--mixed-input-feature-local-only 1`

说明：

- `mixed_input_enabled=1` 时，训练数据集使用 `MixedCutInputStrategy`。
- `input_type_key` 对应 `cut.custom[input_type_key]`，取值应为 `feature` 或 `wav`。
- `feature_local_only=1` 会强制 feature 的 `storage_path` 为本地文件路径（不允许 `s3://`/`tos://` 等 URI）。

## 2. 构建 mixed cuts

新增脚本：

- `egs/france/ASR/local/build_mixed_manifests.py`

可把以下两类输入合并成一个 mixed cuts：

- feature 侧：`--feature-ref-cuts` + `--feature-scp`
- wav 侧：`--wav-cuts` 或 `--wav-jsonl`

示例：

```bash
python egs/france/ASR/local/build_mixed_manifests.py \
  --feature-ref-cuts /path/feature_ref_cuts.jsonl.gz \
  --feature-scp /path/feats.scp \
  --wav-cuts /path/wav_cuts.jsonl.gz \
  --output-cuts /path/train_mixed_cuts.jsonl.gz \
  --weights 1,1 \
  --input-type-key input_type
```

输出中会自动写入：

- `cut.custom[input_type] = "feature"` 或 `"wav"`

## 3. Node Producer 小样本测试（不训练）

建议先做 dataloader 冒烟：

1. 准备少量 mixed cuts（例如 64 条）。
2. 用 `MSR_AsrDataModule + NodeBatchProducer + ConsumerCutSampler` 拉取 2~3 个 step。
3. 验证每个 rank 都能取到 batch，且 batch 内存在 feature/wav 混合样本。

本次在 `h20-6` 的测试产物目录：

- `/data1/mayufeng/tmp/node_mixed_smoke_20260305_145436`
- 结果 metrics：`node_producer_metrics_smoke_ok.jsonl`

## 4. 当前约束（重要）

1. `feature` 输入当前仅支持本地特征路径（`mixed_input_feature_local_only=1`）。
2. 对于 **单个 packed cut 内同时包含 feature + wav** 的情况，当前策略会显式报错（不支持跨类型同 cut 路由）。
3. 如果要稳定验证 node producer + mixed input，建议先使用：
   - `--pack-max-pieces-per-bin 1`

这样可避免把不同输入类型拼进同一个 packed cut。

## 5. 常见错误排查

1. `ModuleNotFoundError: kaldi_native_io`  
   说明环境缺少 kaldi 特征读取依赖；需切换到包含 `lhotse + kaldi_native_io + kaldifeat` 的环境。

2. `feature cut ... does not have features attached`  
   检查 `feature_ref_cuts` 是否正确附带 `features` 字段，并且 `feats.scp` key 可匹配。

3. `mixes multiple input types ['feature', 'wav']`  
   表示同一个 packed cut 内被拼入两种输入类型；先把 `--pack-max-pieces-per-bin` 调为 `1` 进行验证。
