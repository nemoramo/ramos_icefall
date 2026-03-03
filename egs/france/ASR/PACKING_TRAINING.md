# 法语 Zipformer Packing 训练说明

本 recipe 支持 `on-the-fly` 特征 + packing（CutConcatenate），目标是提升 DDP 吞吐并减少 GPU bubble（rank 间 shape/compute 不均导致的等待）。

## 1. 两个“时长”概念（不要混）

- `--max-duration`：DynamicBucketingSampler 用来组成 **一个 batch** 的「原始 raw cuts 总时长上限」（单位秒，per-rank）。
- `--concatenate-cuts-max-duration`：packing 后 **每条 packed cut** 的时长上限（单位秒，比如 30s）。

推荐实践：

- 固定 `--max-cuts`（例如 50）来保证每步 shape 更稳定。
- 固定 `--concatenate-cuts-max-duration 30` 来限制单条 packed cut 的计算量/显存。
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
- `--max-duration >= --max-cuts * --concatenate-cuts-max-duration`：为保证 bestfit/knapsack 下 `--max-duration` 仍能作为每步（per-rank）的严格上界；否则会直接报错。

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
  - `--feature-prefetch-batches 2`
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
- `--wer-lowercase 1`

注意：validation 默认不做 packing（避免改变验证语义）。

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
- `--feature-prefetch-batches`：训练 loop 里提前把未来 N 个 batch 的 `inputs` 搬到 GPU（独立 CUDA stream）。推荐 2；显存吃紧可降到 1 或 0（关闭）。

数据后端（默认 TOS）：

- `--audio-path-backend`：默认 `tos`，会抽样检查 cuts 里音频路径必须以 `--tos-mount-prefix` 开头；本地盘数据用 `local`，调试用 `auto`（跳过检查）。

## 8. 流式 Chunk-based 训练与 packing 的 attention mask 说明

本节主要说明两点：

1) `Zipformer2` 在 chunk-based（causal）训练下生成的 **chunkwise attention mask** 是怎样的；
2) packing 训练时我们构建的 packed block-diagonal mask 会如何与 chunk mask 合并，以及它在流式训练里带来的一个常见“对齐差异”（通常影响很小，但建议记录）。

### 8.1 Zipformer2 的 chunkwise attention mask（非简单下三角）

当 `--causal=1` 且本 step 采样到的 `chunk_size > 0` 时，`Zipformer2` 会生成 chunkwise attention mask：
每个时刻只能看“当前 chunk + 左侧若干 chunks”，而不是简单的下三角 causal mask。

实现位于 `egs/librispeech/ASR/zipformer/zipformer.py` 的 `Zipformer2._get_attn_mask()`：

- 先按时间轴计算 chunk id：`c = t // chunk_size`
- 然后 mask 两类位置（注意：`True=masked`）：
  - `src_c > tgt_c`：未来 chunk（禁止看未来）
  - `src_c < tgt_c - left_context_chunks`：过久远的历史（只保留左侧若干 chunk 的 context）

补充：`chunk_size=-1` 表示 full-context（不会生成 chunk mask）；同时 `chunk_size`/`left_context_frames` 在训练时通常是从列表里随机采样得到的（见 `get_chunk_info()`）。

### 8.2 与 packed block-diagonal mask 的合并逻辑

当启用 `--use-packed-supervisions=1` 且 `--pack-attn-mask=1` 时，packing 训练会为 packed batch 构造
packed block-diagonal external mask（不同 supervision 段之间互相全禁，常见形状为 `(B,T,T)`，`True=masked`）。

在 `Zipformer2.forward()` 中，external `attn_mask` 会与内部的 `chunk_attn_mask` 通过 `torch.logical_or` 合并
（必要时会把 2D chunk mask unsqueeze 成 3D 以便与 per-batch mask 广播）。

由于 `True=masked`，`logical_or` 等价于：
- masked 集合取并集
- allowed 集合取交集

也就是说：**既要满足 chunk 约束，也要满足 block-diagonal 约束**（两者同时生效）。

### 8.3 packing 对 chunk 边界对齐的影响（随机 chunk offset）

你可能会注意到一个“训练 vs 推理”的对齐差异：

- 不 packing 时：每条 utterance 都从 `t=0` 开始，所以 chunk 边界天然对齐到 utterance 开头（末尾 chunk 可能不满，这两者都一样）。
- packing 后：chunk mask 的 chunk 划分是按 packed 序列的绝对时间 `t=0..T-1` 来切的；而每个 supervision 段的起点 `s`
  往往不在 `chunk_size` 的整数倍上，于是该 utterance 的“第一个 chunk”相当于带了一个随机 offset（开头可能是一个“残 chunk”）。

这会导致：训练时看到的 chunk 对齐方式与推理时“每条 utterance 从 0 开 streaming”不完全一致。
通常影响很小，更像是一种 **random chunk-offset augmentation**（随机 chunk 对齐增强），不需要为了“严格对齐”专门改训练实现；
但建议在实验记录里明确写清楚是否启用了 packing、`--causal`、`--chunk-size`/`--left-context-frames` 等配置，方便复现实验与排障对比。

### 8.4 当前 master 的 mask 长度对齐修复

近期 `master` 已修复 Zipformer 中一个常见的 mask 长度不匹配问题（典型触发场景：external packed mask 与 chunk mask/attention score 在尾部长度上存在 off-by-one 差异）：

- 在 `Zipformer2.forward()` 合并 `attn_mask` 与 `chunk_attn_mask`（`logical_or`）之前，先做维度对齐（必要时 pad/crop）。
- 在 `RelPositionMultiheadAttentionWeights.forward()` 的 `masked_fill(attn_mask, ...)` 之前，先把 `attn_mask` 对齐到 `attn_scores` 形状。

这属于鲁棒性修复，不改变正常 shape 完全匹配时的训练语义。

## 9. Node Producer（节点级数据生产者）

Node Producer 是一种用于大规模 DDP 训练的**数据预取与分发架构**，通过在生产者-消费者模式中将数据打包逻辑集中到 node 级别的独立线程，解决传统 DataLoader 每个 rank 独立采样导致的负载不均和 GPU bubble 问题。

### 9.1 核心原理

#### 架构对比

**传统 DDP DataLoader（无 Node Producer）：**
```
Rank 0: [Sampler] -> [CutConcatenate] -> [DataLoader Worker] -> [GPU]
Rank 1: [Sampler] -> [CutConcatenate] -> [DataLoader Worker] -> [GPU]
Rank 2: [Sampler] -> [CutConcatenate] -> [DataLoader Worker] -> [GPU]
Rank 3: [Sampler] -> [CutConcatenate] -> [DataLoader Worker] -> [GPU]
```
问题：每个 rank 独立采样和 packing，容易造成各 rank 间 batch shape 差异大，导致快的 rank 等待慢的 rank（bubble）。

**Node Producer 架构：**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Node 0 (4 GPUs)                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Node Batch Producer (线程)                        │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │ PackAwareDistributedDynamicBucketingSampler                   │ │   │
│  │  │  - 统一采样和 packing                                          │ │   │
│  │  │  - next_rank_splits() 生成每 rank 的数据分割                    │ │   │
│  │  └───────────────────────────┬───────────────────────────────────┘ │   │
│  │                              ▼                                      │   │
│  │  ┌──────────┬──────────┬──────────┬──────────┐                     │   │
│  │  │ Queue 0  │ Queue 1  │ Queue 2  │ Queue 3  │  ← 每 rank 独立队列   │   │
│  │  │ (size=32)│ (size=32)│ (size=32)│ (size=32)│                     │   │
│  │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘                     │   │
│  └───────┼──────────┼──────────┼──────────┼─────────────────────────────┘   │
│          │          │          │          │                                  │
│  ┌───────▼──────────▼──────────▼──────────▼───────┐                          │
│  │       ConsumerCutSampler (各 rank 独立)         │                          │
│  │  Rank 0  Rank 1  Rank 2  Rank 3                │                          │
│  └───────┬──────────┬──────────┬──────────┬───────┘                          │
│          │          │          │          │                                  │
│  ┌───────▼──────────▼──────────▼──────────▼───────┐                          │
│  │                   GPUs                         │                          │
│  └────────────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 关键组件

1. **NodeBatchProducer** (`node_batch_producer.py`)
   - 运行在独立的 daemon 线程中
   - 使用 `PackAwareDistributedDynamicBucketingSampler` 统一采样和 packing
   - 通过 `next_rank_splits()` 生成每个 rank 的 cuts 分割
   - 将打包好的 batch 推送到各 rank 的 IPC 队列

2. **NodeBatchIPC** (`node_batch_ipc.py`)
   - 使用 `multiprocessing.Manager()` 创建共享队列和事件
   - `rank_queues`: 每个 rank 的 `Manager.Queue(maxsize=queue_size)`
   - `stop_event`: 跨进程停止信号
   - `producer_error`: 生产者错误信息广播
   - `metrics`: 共享字典记录生产和消费进度

3. **ConsumerCutSampler** (`consumer_sampler.py`)
   - 每个 rank 的 `DataLoader` 使用该采样器
   - 从对应 rank 的队列中 `get()` 数据
   - 处理 `epoch_end` 信号来结束当前 epoch
   - 记录消费进度和等待时间到共享 metrics

#### 数据流向

```
1. Sampler 从 manifest 中采样 cuts
        ↓
2. CutConcatenate 进行 packing（bestfit/knapsack）
        ↓
3. next_rank_splits() 将 packed cuts 按 rank 分割
        ↓
4. 每个 rank 的 cuts 包装成 {"type": "batch", "cuts": cuts, "step_id": N}
        ↓
5. _put_blocking() 推送到对应 rank 的 Queue
        ↓
6. ConsumerCutSampler 从 Queue get() 数据
        ↓
7. OnTheFlyFeatures 实时提取 Fbank 特征
        ↓
8. _iter_with_feature_prefetch 异步拷贝到 GPU
```

### 9.2 参数详解

#### 启用与基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--node-data-producer` | bool | False | **总开关**：是否启用 node producer 模式（仅 DDP 有效） |
| `--node-data-producer-queue-size` | int | 32 | 每 rank 的队列长度（buffers）。越大越能平滑生产波动，但占用更多内存 |
| `--node-data-producer-max-ahead-steps` | int | 16 | 预留参数（当前由 queue_size 控制实际深度） |

#### 消费者行为

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--node-data-producer-block-on-empty` | bool | True | 消费者队列为空时是否阻塞等待。True 保证训练稳定性；False 用于特殊调试场景 |
| `--node-data-producer-block-timeout-sec` | int | 0 | 消费者等待超时（秒），<=0 表示无限等待。用于检测 producer 卡死或异常 |

#### 生产者监控

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--node-data-producer-log-interval` | int | 20 | Producer 日志记录间隔（秒），记录 produced_steps、queue_depth、lag_steps 等 |
| `--node-data-producer-heartbeat-sec` | int | 2 | Producer 心跳更新间隔（秒），用于计算 produced_steps_per_sec 速率 |
| `--node-data-producer-metrics-out` | str | "" | Metrics JSONL 输出路径，默认 `<exp_dir>/node_producer_metrics.jsonl` |

### 9.3 启用 Node Producer

在 `run_french_150M.sh` 中启用（需要 DDP，world_size > 1）：

```bash
cd egs/france/ASR
NODE_DATA_PRODUCER=1 ./run_french_150M.sh
```

或完整参数覆盖示例：

```bash
NODE_DATA_PRODUCER=1 \
NODE_DATA_PRODUCER_QUEUE_SIZE=48 \
NODE_DATA_PRODUCER_LOG_INTERVAL=10 \
./run_french_150M.sh
```

**注意：**
- Node Producer 仅在 DDP 模式（world_size > 1）下有效
- Rank 0 负责运行 Producer 线程，其他 rank 作为消费者
- 启动时会有 barrier 同步，确保 producer 就绪后各 rank 才开始训练

### 9.4 可视化监控（Dashboard）

Node Producer 会生成 `node_producer_metrics.jsonl`，可以通过轻量 HTTP 服务实时监控：

**启动 Dashboard：**

```bash
cd egs/france/ASR
python local/node_producer_dashboard.py \
  --metrics-file /path/to/exp/node_producer_metrics.jsonl \
  --host 0.0.0.0 \
  --port 8787 \
  --queue-size 32
```

**访问方式：**

```bash
http://<机器IP>:8787
```

**监控面板指标说明：**

- **Acquire (Fetch)**：`produced_steps_per_sec`，producer 实时产出速率
- **Process (Pack)**：`process_rate_steps_per_sec`，有效产出速率（基于历史斜率）
- **Consume (Ranks)**：`consume_rate_steps_per_sec`，各 rank 中最慢的消费速率
- **Queue Depth**：每个 rank 队列的积压深度（实时）
- **Lag Steps**：各 rank 消费进度的最大差异（用于检测负载不均）
- **Backlog Steps**：每个 rank 相对于 producer 的落后 batch 数

**Dashboard 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--history-limit` | 1200 | 服务端内存中保留的最大记录数 |
| `--history-points` | 320 | 前端每次拉取的历史数据点数（用于绘图） |
| `--refresh-ms` | 1000 | 前端自动刷新周期（毫秒） |
| `--queue-size` | 32 | 用于计算 queue fill ratio（应与训练时一致） |

### 9.5 性能调优建议

#### Queue Size 选择

```
queue_size = 32  (默认，适合大多数场景)
queue_size = 48  (I/O 波动大或 packing 耗时长时)
queue_size = 16  (内存紧张，接受轻微 stall)
```

**判断标准：** 观察 dashboard 中的 `queue_fill_ratio`：
- 长期 < 30%：queue_size 可能过大，浪费内存
- 频繁降到 0%：queue_size 太小，producer 跟不上，需要增大或优化采样效率

#### 处理 Producer 滞后

如果发现 `lag_steps` 持续增大（consumer 比 producer 慢）：

1. **检查 packing 效率**：增大 `--pack-raw-pool-size`（如 12000/20000）
2. **增加 I/O 并发**：增大 `--num-workers` 和 `--prefetch-factor`
3. **降低单个 batch 的 packing 复杂度**：减小 `--pack-max-pieces-per-bin`

#### 与其他参数的交互

| 相关参数 | 与 Node Producer 的关系 |
|----------|------------------------|
| `--packing-method knapsack` | Node Producer 配合 knapsack packing 效果最佳，rank 间负载最均匀 |
| `--max-cuts` | 固定 max-cuts 是 Node Producer 高效工作的前提，保证每 step shape 稳定 |
| `--feature-prefetch-batches` | Node Producer 负责 CPU->CPU 的 batch 预取，feature_prefetch_batches 负责 CPU->GPU 的异步拷贝，两者是正交叠加的 |

### 9.6 Consumer Batch Replay（消费者批次回放）

当 Producer 生产速度暂时跟不上 Consumer 消费速度时（如 I/O 抖动、packing 计算波动），传统做法是阻塞等待或超时报错，导致 GPU 空闲浪费算力。**Batch Replay** 机制允许 Consumer 在历史 batch 中随机选择并回放，同时保持 GPU 持续训练。

#### 工作原理

```
Consumer 尝试从 Queue 取数据
        ↓
    Queue 为空 → 开始计时等待
        ↓
    等待超过阈值 (如 500ms)
        ↓
    检查 replay 触发条件（多层级防护）
        ↓
    从最近 N 个 batch 中随机选择一个
        ↓
    进行数据增强（shuffle cuts + shuffle tracks）
        ↓
    作为 "replay_batch" 返回给 GPU
```

**Replay 触发条件（必须全部满足）：**

| 条件 | 默认值 | 说明 |
|------|--------|------|
| `replay_on_empty` | False | 功能总开关 |
| 等待时间 | 500ms | 必须超过 `replay_wait_threshold_ms` |
| 缓冲区非空 | - | 最近 batch 缓存区有数据 |
| 最小间隔步数 | 100 steps | 两次 replay 间隔至少 100 个正常 batch |
| 全局比例上限 | 3% | replay 样本占当前 epoch 总样本比例上限 |
| 概率抽样 | 25% | 满足以上条件后，以 `replay_prob` 概率触发 |

**Replay 数据增强：**

回放的 batch 不是原样返回，而是进行轻微数据增强，相当于提供"新的"训练样本：

1. **Shuffle packed-cut 顺序**：改变 batch 内各 packed cut 的顺序
2. **Shuffle tracks（utterance 顺序）**：对每个 MixedCut，打乱内部拼接的 utterance 顺序，但保持原始 gap 间隔

示例：
- 原 batch: `[cut_A(3条语音拼接), cut_B(2条语音拼接)]`
- Replay 后: `[cut_B'(内部2条语音顺序打乱), cut_A'(内部3条语音顺序打乱)]`

#### Replay 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--node-data-producer-replay-on-empty` | bool | False | **总开关**：是否启用 batch replay |
| `--node-data-producer-replay-wait-threshold-ms` | float | 500.0 | 触发 replay 的等待时间阈值（毫秒）|
| `--node-data-producer-replay-buffer-size` | int | 8 | 保存最近 batch 的缓冲区大小 |
| `--node-data-producer-replay-prob` | float | 0.25 | 满足条件时触发 replay 的概率 |
| `--node-data-producer-replay-min-interval-steps` | int | 100 | 两次 replay 之间的最小步数间隔 |
| `--node-data-producer-replay-max-ratio` | float | 0.03 | 当前 epoch 中 replay 占总样本的最大比例（0.03=3%）|

#### 启用与配置示例

**基础启用：**
```bash
cd egs/france/ASR
NODE_DATA_PRODUCER_REPLAY_ON_EMPTY=1 ./run_french_150M.sh
```

**激进配置（Producer 严重滞后场景）：**
```bash
NODE_DATA_PRODUCER_REPLAY_ON_EMPTY=1 \
NODE_DATA_PRODUCER_REPLAY_WAIT_THRESHOLD_MS=200 \
NODE_DATA_PRODUCER_REPLAY_PROB=0.5 \
NODE_DATA_PRODUCER_REPLAY_MAX_RATIO=0.05 \
NODE_DATA_PRODUCER_REPLAY_MIN_INTERVAL_STEPS=50 \
./run_french_150M.sh
```

**保守配置（仅极端情况才 replay）：**
```bash
NODE_DATA_PRODUCER_REPLAY_ON_EMPTY=1 \
NODE_DATA_PRODUCER_REPLAY_WAIT_THRESHOLD_MS=1000 \
NODE_DATA_PRODUCER_REPLAY_PROB=0.1 \
NODE_DATA_PRODUCER_REPLAY_MAX_RATIO=0.01 \
./run_french_150M.sh
```

#### 使用建议与注意事项

**何时启用：**
- 观察到训练过程中有频繁的 `consumer_wait_ms` 尖峰（通过 dashboard）
- GPU util 不稳定，出现周期性下降
- Producer 因 I/O 波动或 packing 计算量波动而偶发卡顿

**监控指标：**
- 通过 dashboard 观察各 rank 的 `consumer_replay_count_per_rank`
- 如果某个 rank 的 replay 频率持续高于 5%，说明 producer 瓶颈严重，应从源头解决：
  1. 增大 `--node-data-producer-queue-size`
  2. 优化 packing 参数（增大 `--pack-raw-pool-size`）
  3. 增加 I/O 并发（增大 `--num-workers` 和 `--prefetch-factor`）

**注意事项：**
- replay 相当于数据增强，理论上不会损害模型质量
- 但 replay 的样本没有新的音频数据，只是重新排列组合已有样本
- replay 比例应控制在 5% 以内，过高则说明需要优化 producer 性能
- replay 会略微增加 CPU 开销（shuffle cuts 和 tracks），但通常可忽略

### 9.7 故障排查

**Producer 卡死/无输出：**
```bash
# 检查 metrics 文件是否有更新
tail -f exp/xxx/node_producer_metrics.jsonl

# 查看 queue depth 是否一直为 0
grep "queue_depth" exp/xxx/node_producer_metrics.jsonl | tail

# 查看是否有 replay 计数在增加（说明 consumer 在等数据）
grep "consumer_replay_count" exp/xxx/node_producer_metrics.jsonl | tail
```

**Consumer 超时：**
```bash
# 增大超时时间或检查 producer 是否报错
NODE_DATA_PRODUCER_BLOCK_TIMEOUT_SEC=60 ./run_french_150M.sh
```

**某个 rank 消费明显慢于其他 rank：**
- 检查该 rank 的 GPU util（可能是模型 forward 有瓶颈）
- 检查 `consumer_wait_ms_per_rank` 是否异常高（可能是该 rank 的 DataLoader 进程问题）
- 检查 `consumer_replay_count_per_rank` 是否明显高于其他 rank（该 rank 经常"饿"）

## 10. 与当前 launcher 对齐的最小启动参数（2026-03）

`run_french_150M.sh` 当前默认关键参数如下（可按需覆盖）：

- `PACKING_METHOD=knapsack`
- `MAX_DURATION=1500`
- `MAX_CUTS=50`
- `NUM_WORKERS=16`
- `TRAIN_PREFETCH_FACTOR=16`
- `FEATURE_PREFETCH_BATCHES=2`
- `AUDIO_PATH_BACKEND=tos`
- `COMPUTE_VALID_WER=1`
- `VALID_INTERVAL=1000`

示例（不改脚本，临时覆盖）：

```bash
cd egs/france/ASR
FEATURE_PREFETCH_BATCHES=2 MAX_DURATION=1800 ./run_french_150M.sh
```
