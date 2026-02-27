# 法语 Zipformer Packing 训练说明

本 recipe 支持 `on-the-fly` 特征 + packing（CutConcatenate），目标是提升 DDP 吞吐并减少 GPU bubble（rank 间 shape/compute 不均导致的等待）。

## 1. 两个“时长”概念（不要混）

- `--max-duration`：DynamicBucketingSampler 用来组成 **一个 batch** 的「原始 raw cuts 总时长上限」（单位秒，per-rank）。
- `--concatenate-cuts-max-duration`：packing 后 **每条 packed cut** 的时长上限（单位秒，比如 30s）。

推荐实践：

- 固定 `--max-cuts`（例如 50）来保证每步 shape 更稳定。
- 固定 `--concatenate-cuts-max-duration 30` 来限制单条 packed cut 的计算量/显存。
- `bestfit/knapsack` 模式下需满足：`--max-duration >= --max-cuts * --concatenate-cuts-max-duration`（否则会直接报错）。
- 再通过扫 `--max-duration`（例如 1500 -> 1800 -> 2100 -> 2400）把显存/利用率推满。

## 2. Packing 方法（简化参数）

使用 `--packing-method` 一键切换三种模式，避免一堆低层 flag 组合：

- `none`：不做 concatenation/packing。
- `lhotse_legacy`：dataset 侧做 `CutConcatenate`（DDP 下每个 rank 独立 pack，容易产生 bubble）。
- `bestfit`：DDP pack sampler + raw best-fit 装箱。
- `knapsack`：DDP pack sampler + best-fit + 尾部 knapsack（推荐）。

对应关系（内部会覆盖低层开关）：

- `lhotse_legacy`：
  - `--concatenate-cuts 1`
  - `--ddp-pack-sampler 0`
- `bestfit`：
  - `--concatenate-cuts 1`
  - `--ddp-pack-sampler 1`
  - `--pack-fill-strategy raw_best_fit`
- `knapsack`：
  - `--concatenate-cuts 1`
  - `--ddp-pack-sampler 1`
  - `--pack-fill-strategy raw_best_fit_knapsack`

约束（bestfit/knapsack 必须满足，否则直接报错）：

- `--max-cuts > 0`：需要固定每 rank 的 packed cut 数，才能让每步 shape 稳定。
- `--concatenate-cuts-max-duration > 0`：需要明确每条 packed cut 的上限（秒）。
- `--max-duration >= --max-cuts * --concatenate-cuts-max-duration`：raw_best_fit* 下每步输出条数主要由 `max_cuts` 决定；该约束保证 `max-duration` 仍然是严格上界（避免 OOM/语义混乱）。

## 3. 推荐默认参数（当前 france recipe）

`MSR_AsrDataModule` 默认值已按 packing 训练调过（与 phaseC 对齐）：

- Packing：
  - `--packing-method knapsack`
  - `--ddp-pack-sampler 1`
  - `--concatenate-cuts 1`
  - `--concatenate-cuts-max-duration 30`
  - `--gap 1.0`
  - `--use-packed-supervisions 1`
  - `--pack-attn-mask 1`
  - `--pack-raw-pool-size 8000`
  - `--pack-max-pieces-per-bin 10`
  - `--pack-min-remaining-duration 0.5`
  - `--pack-tail-knapsack-rem 5.0`
  - `--pack-tail-knapsack-max-candidates 128`
  - `--pack-tail-knapsack-max-pieces 4`
- Dynamic bucketing：
  - `--max-duration 1500`
  - `--max-cuts 50`
  - `--num-buckets 60`
  - `--bucketing-buffer-size 30000`
  - `--bucketing-shuffle-buffer-size 30000`
- DataLoader：
  - `--num-workers 16`
  - `--prefetch-factor 16`
  - `--valid-num-workers 6`
  - `--valid-prefetch-factor 8`

## 4. 数据读取后端（默认：TOS）

为避免 NFS 高并发拥挤，本 recipe 默认要求使用 TOS mount 路径：

- `--audio-path-backend tos`（默认）
- `--tos-mount-prefix /mnt/asr-audio-data`（默认）

行为说明：

- `tos`：从每个 cuts manifest 里抽样 N 条 cut，发现音频路径不在该 prefix 下就 **fail fast**。
- `local`：允许本地路径（不做检查）。
- `auto`：不做后端检查（只建议调试用）。

如果你的音频数据就是本地盘（非 /mnt/asr-audio-data），请显式用 `--audio-path-backend local`。

## 5. 运行方式

推荐直接跑 launcher：

```bash
cd egs/france/ASR
./run_french_150M.sh
```

常用覆盖方式：

```bash
# 本地音频路径（跳过 TOS prefix 强校验）
AUDIO_PATH_BACKEND=local ./run_french_150M.sh

# 扫 max-duration 推显存/利用率
MAX_DURATION=1800 ./run_french_150M.sh

# 对比 legacy packing（更容易 bubble）
PACKING_METHOD=lhotse_legacy ./run_french_150M.sh
```

## 6. Validation / WER

长跑建议（成本可控）：

- `--valid-interval 1000`
- `--compute-valid-wer 1`
- `--valid-wer-max-batches 100`
- `--valid-error-rate wer`（可选：`cer`）
- `--wer-lowercase 1`

注意：validation 默认不做 packing（避免改变验证语义）。
如果你显式打开了 packing（例如 `--valid-concatenate-cuts 1`），validation 仍可算 loss，但会跳过 WER（greedy-search 路径假设 1-utt-per-seq）。

## 7. 关键参数说明与调参建议

Packing 相关：

- `--packing-method`：推荐 `knapsack`（别名 `ksnapbak`），其次 `bestfit`，对比用 `lhotse_legacy`。
- `--use-packed-supervisions`：必须为 1 才能把 packed cut 里的每个 supervision 当作独立 utterance 来算 RNNT/CTC（不是把整条 packed cut 当一句）。
- `--pack-attn-mask`：是否构造 packed 的 attention mask（避免注意力跨 utterance）。开了更稳，但会增加显存/计算开销；显存压力大时可先关再对比 WER。
- `--concatenate-cuts-max-duration`：单条 packed cut 上限（秒）。推荐 30；调大更吃显存，调小会降低吞吐。
- `--gap`：packed cut 内 utterance 之间插入的间隔（秒）。越大越“干净”但越浪费计算；推荐先用 1.0，后续可尝试 0.1/0.2 做对比。

Bestfit/knapsack 装箱质量与均匀性：

- `--pack-raw-pool-size`：raw-cut 缓冲池大小。越大越容易把每条 packed cut 填满，rank 间计算更均匀，但 CPU/memory 会更高。推荐 8000；想再压 padding 可上调到 12000/20000。
- `--pack-max-pieces-per-bin`：每条 packed cut 最多拼多少段。越大越容易填满，但 supervision 段数多会增加 loss/对齐开销。推荐 10。
- `--pack-min-remaining-duration`：剩余时长小于该阈值就停止填充。越小越“抠”，填充率更高但 CPU 开销更大。推荐 0.5。
- `--pack-tail-knapsack-rem`：当剩余时长 <= 该阈值时触发尾部 knapsack（只在 `knapsack` 模式有效）。推荐 5.0。
- `--pack-tail-knapsack-max-candidates`：尾部 knapsack 的候选短句数量上限。越大越容易补洞，但 CPU 更重。推荐 128。
- `--pack-tail-knapsack-max-pieces`：尾部 knapsack 最多再补几段。推荐 4。

Batch 形状与显存利用：

- `--max-cuts`：每 rank 每步的 packed cut 条数（固定它是 bestfit/knapsack 的前提）。推荐 50；想要更高吞吐可以加大，但会更吃显存。
- `--max-duration`：每 rank 的 pooled raw 时长上限（秒）。推荐从 1500 起步，根据显存与 util 逐档上调（1800/2100/2400/3000）。

Sampler 混洗与 I/O 并发：

- `--bucketing-buffer-size` / `--bucketing-shuffle-buffer-size`：越大混洗越充分，packing 更容易凑满；推荐 30000。
- `--num-workers` / `--prefetch-factor`：提高可减小 data stall；推荐 16/16（valid/test 8）。如果 CPU 充足且 I/O 稳定，可继续上调做对比。

数据后端（默认 TOS）：

- `--audio-path-backend`：默认 `tos`，会抽样检查 cuts 里音频路径必须以 `--tos-mount-prefix` 开头；本地盘数据用 `local`，调试用 `auto`（跳过检查）。

## 8. 流式 Chunk-based 训练与 packing 的 attention mask 说明

1) 当 `chunk_size != -1` 时，`Zipformer2` 会生成 **chunkwise attention mask**（不是简单的下三角 causal），其约束是“只能看当前 chunk + 左侧若干 chunks”：

- `egs/librispeech/ASR/zipformer/zipformer.py` 里满足 `src_c > tgt_c`（未来 chunk）或 `src_c < tgt_c - left_context_chunks`（太久远的历史）的注意力会被 mask 掉。

2) packing 训练时（CutConcatenate），我们构建的是 **packed block-diagonal mask**（不同 supervision 段之间互相全禁）。它会与 chunk mask 做 `logical_or` 合并：

- `True = masked`，因此 OR 等价于“允许集合取交集”：既要满足 chunk 约束，又要满足 block-diagonal 约束。

3) chunk 边界与每条 utterance 的对齐方式在 packing 与非 packing 下不完全一致：

- chunk mask 的 chunk 划分按 packed 序列的绝对时间 `t=0..T-1` 来切；
- packing 后每个 supervision 段起点 `s` 往往不在 `chunk_size` 的整数倍上，因此“这条 utterance 的第一个 chunk”会带一个随机 offset（开头可能是一个“残 chunk”）；
- 不 packing 时，每条 utterance 都从 `t=0` 开始，所以 chunk 边界天然对齐到 utterance 开头（末尾 chunk 仍可能不满，两者都一样）。

通常这类差异影响很小，更像一种 random chunk offset augmentation；建议在实验记录中说明 packing 是否开启以及对应的 chunk 参数即可。
