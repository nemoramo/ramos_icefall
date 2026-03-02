# France 法语 ASR (Zipformer) 说明

本目录用于大规模法语 ASR 训练，特征采用 **on-the-fly** 提取（Lhotse `OnTheFlyFeatures` + Fbank）。

在高并发训练场景（8 卡 * 多 dataloader workers）下，如果音频都在共享 NFS 盘上，容易出现 I/O 拥挤、抖动甚至卡死。更推荐的方式是使用 TOS bucket `asr-audio-data` 的 fsx/hpvs FUSE 挂载（例如 `/mnt/asr-audio-data`）来读取音频，以提升稳定性。

## 入口脚本

- 训练入口: `egs/france/ASR/run_french_150M.sh`
- 训练主程序: `egs/france/ASR/zipformer/train.py`

## 数据约定

`run_french_150M.sh` 默认期望以下文件：

- `MANIFEST_DIR`: Lhotse cuts（JSONL.GZ）所在目录：
  - `msr_cuts_French_train.jsonl.gz`
  - `msr_cuts_French_valid.jsonl.gz`
  - `msr_cuts_French_test.jsonl.gz`
- `BPE_MODEL`: 法语 SentencePiece 模型（默认 `lang_bpe_2048/bpe.model`）

cuts 中的音频路径必须在训练机器上可读。

## 使用 TOS 挂载提升 I/O 稳定性

### 快速检查是否已挂载

```bash
findmnt /mnt/asr-audio-data
grep -F " /mnt/asr-audio-data " /proc/mounts
```

### 挂载示例（fsx/hpvs fusedaemon）

```bash
sudo /opt/fsx/tools/start_fusedaemon.sh \
  --fusedaemon_combination_name=tos \
  --fusedaemon_support_instance_list=tos

sudo mkdir -p /mnt/asr-audio-data
sudo mount -t fsx /tos/asr-audio-data /mnt/asr-audio-data \
  -o region="ap-southeast-1",endpoint="https://tos-ap-southeast-1.ivolces.com",credential_filepath="/opt/credential.json",no_writeback_cache

findmnt /mnt/asr-audio-data
```

注意：
- 不要把任何 AK/SK、credential 文件内容提交到 git。
- mount endpoint 常见是 `tos-<region>...`；S3 兼容 API endpoint 常见是 `tos-s3-<region>...`。

## 将 cuts 音频路径切换到挂载前缀

如果你的 cuts 里还是 NFS 前缀，可以用脚本把 `recording.sources[].source` 的前缀替换为 `/mnt/asr-audio-data`：

```bash
python egs/france/ASR/local/replace_cut_source_prefix.py \
  --input-cuts  "${MANIFEST_DIR}/msr_cuts_French_train.jsonl.gz" \
  --output-cuts "${MANIFEST_DIR}/msr_cuts_French_train.tos.jsonl.gz" \
  --src-prefix  "/nfs/audio_root/" \
  --dst-prefix  "/mnt/asr-audio-data/audio_root/"
```

## 大规模数据：绕开 `lhotse kaldi import`

对于特别大的数据集，`lhotse kaldi import` 在 I/O 拥挤时可能会非常慢甚至卡住。这里提供了一个流式转换脚本：

- `egs/france/ASR/local/json_to_lhotse_cuts_fast.py`

它可以把包含 `audio_filepath` + `duration` (+ 文本字段) 的 JSONL 转为 Lhotse cuts JSONL.GZ。

### 混合外部 JSONL 数据（含随机混合）

如果需要把外部 jsonl（例如：
`/mnt/asr-audio-data/users/yufeng.ma/lemas_dataset/manifests/train/en/en_all.nemo.wav16k.jsonl`）
与现有 train cuts 混合，建议分两步：

1. 先把外部 jsonl 转成 cuts；
2. 再用流式随机混合脚本合并到新的 train cuts。

新增脚本：

- `egs/france/ASR/local/mix_cuts_manifests.py`

示例（建议默认）：

```bash
# 1) 外部 jsonl -> cuts
python egs/france/ASR/local/json_to_lhotse_cuts_fast.py \
  --input-json /mnt/asr-audio-data/users/yufeng.ma/lemas_dataset/manifests/train/en/en_all.nemo.wav16k.jsonl \
  --output-cuts /path/to/manifests/lemas_en_train_cuts.jsonl.gz \
  --id-prefix lemas-en- \
  --text-field original_text \
  --fallback-text-field text \
  --text-norm none \
  --max-duration 30 \
  --drop-empty-text 1

# 2) 与现有 train cuts 随机混合（权重示例：原始:新增=3:1）
python egs/france/ASR/local/mix_cuts_manifests.py \
  --input-cuts /path/to/manifests/current_train_cuts.jsonl.gz /path/to/manifests/lemas_en_train_cuts.jsonl.gz \
  --weights 3,1 \
  --output-cuts /path/to/manifests/train_cuts_mixed_v1.jsonl.gz \
  --buffer-size 20000 \
  --seed 777 \
  --max-duration 30 \
  --attach-source-tag 1
```

文本字段建议：

- 优先使用 `original_text`，回退到 `text`；
- 默认 `--text-norm none`，先不做统一小写/去标点，避免和现有数据风格不一致；
- 如果后续验证发现不稳定，再单独试 `lower` 或 `lower_no_punc` 做对照实验。

随机性说明：

- `mix_cuts_manifests.py` 在构建 manifest 阶段做一次随机混合（可复现，受 `--seed` 控制）；
- 训练阶段 `DynamicBucketingSampler` 还会继续按 buffer 做 shuffle，因此整体随机性是叠加的。

## I/O 基准：FUSE 挂载 vs 直连 TOS API

用于对比：

- 从本地挂载路径读取并解码（FUSE）
- 通过 TOS S3 兼容 API（boto3 `get_object`）直接拉取并解码（TOS）

脚本：

- `egs/france/ASR/local/benchmark_french_tos_vs_fuse.py`

如需使用 TOS API 模式，需要配置环境变量（或传入 `.env` 文件）：

- `TOS_ACCESS_KEY_ID`
- `TOS_SECRET_ACCESS_KEY`
- `TOS_ENDPOINT`（S3 兼容 endpoint，例如 `https://tos-s3-<region>.ivolces.com`）
- `TOS_REGION`
