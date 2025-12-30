# Tools

## `force_align.py`: 强制对齐（Forced Alignment）

给定一条音频 + 参考文本（reference transcript），输出**每个词**在音频中的对齐时间（默认输出 start；`--with-end` 可输出 start-end）。

这个工具支持**同时输入多个 `.pt` checkpoint**，并按 checkpoint 自动切换 forward 方式：

- `causal=false`（非流式模型）：走 offline 全量前向 `model.forward_encoder()`
- `causal=true`（流式模型）：**必须走真流式** `encoder_embed.streaming_forward()` + `Zipformer2.streaming_forward()`（带状态、按 chunk 推理）
  - chunk 参数 `chunk_size / left_context_frames` 默认从 `.pt` 中读取并选择（可用 YAML/CLI 覆盖，但必须在 `.pt` 列表内）

对齐头（alignment head）可选：

- `--align-kind ctc`（默认）：使用 CTC 头强制对齐；可输出 token duration，因此 `--with-end` 有效
- `--align-kind rnnt`：使用 Transducer/RNNT 头强制对齐；当前实现只提供 token start time（没有 duration），因此 `--with-end` 只会显示 start
- `--align-kind auto`：如果模型包含 RNNT 头则优先用 `rnnt`，否则回退到 `ctc`

### 为什么必须区分 streaming/offline forward？

Zipformer 的 “模拟流式”（attention mask + 随机 chunk）和 “真流式”（`streaming_forward()` + states）**不是同一条计算路径**。  
如果要评估/复现流式系统的时间戳行为，`causal=true` 的模型必须用 `streaming_forward()` 跑出来的 encoder_out 再做对齐，否则会产生不一致的时间偏移。

### 原理（实现流程）

以强制对齐为例（`--align-kind ctc|rnnt|auto`）：

1. 读入 wav（`torchaudio.load`），必要时重采样到 16k（`--sample-rate`）。
2. 提取 80 维 fbank（`torchaudio.compliance.kaldi.fbank`，10ms frame shift）。
3. 计算 encoder 输出：
   - offline：一次性 `forward_encoder(features)`
   - streaming：按 chunk 输入 fbank，维护 encoder/cache states，累计拼接成完整 `encoder_out`
4. 调用 `icefall.forced_alignment.force_align(kind=...)`：
   - CTC：用 `model.ctc_output(encoder_out)` 得到 log-probs，再用 `torchaudio.functional.forced_align` 做对齐（可得到 duration）
   - RNNT：用模型的 `decoder/joiner` 做对齐（当前实现输出 token start time，没有 duration）
   - 最终都用 subsampling factor + frame shift 把 frame index 换算为秒
5. 把 SentencePiece token 合并成词：
   - 依据 token 的 `▁` 词边界标记进行分组
   - 通过 `SentencePieceProcessor.decode_pieces()` 得到最终的词字符串

注意：时间戳在 subsampled frame 上是量化的。常见设置 `frame_shift=10ms, subsampling_factor=4` => 时间步长约 `0.04s`。

### 输入格式

`tools/force_align.py` 支持三种输入（任选其一）：

1. **manifest（JSONL）**
   - 每行至少包含：`audio_filepath`, `text`, `utt_id`
2. **Kaldi 风格**
   - `wav.scp` + `text`
   - 出于安全考虑：不支持 `wav.scp` 中的管道（以 `|` 结尾的命令）
3. **单条音频**
   - `--audio /path/to.wav --transcript "..."`（可以用于临时验证）

### 典型用法

#### 1) manifest + 多个 checkpoint（每个模型一行）

```bash
python3 tools/force_align.py \
  --recipe-dir egs/spgispeech/ASR/zipformer \
  --ckpt /path/to/stream.pt \
  --ckpt /path/to/offline.pt \
  --name stream150 --name offline_ep4 \
  --bpe-model /path/to/bpe.model \
  --manifest /path/to/manifest.jsonl \
  --utt-id 0 \
  --with-end
```

输出示例（同一个 utt_id 下，每个 checkpoint 一行）：

```text
utt_id=0 unk=0 audio=/data1/.../12.wav
text=daniel how do we think ...
stream150 DANIEL@0.00-1.08 HOW@1.28-1.36 ...
offline_ep4 DANIEL@0.00-0.92 HOW@1.04-1.12 ...
```

#### 2) wav.scp + text（Kaldi）

```bash
python3 tools/force_align.py \
  --recipe-dir egs/spgispeech/ASR/zipformer \
  --ckpt /path/to/model.pt \
  --bpe-model /path/to/bpe.model \
  --wav-scp data/wav.scp \
  --text data/text \
  --utt-id utt000001 \
  --with-end
```

#### 3) YAML 配置（可选）

```yaml
recipe_dir: egs/spgispeech/ASR/zipformer
device: cuda:0
bpe_model: /path/to/bpe.model
ckpts:
  - /path/to/stream.pt
  - /path/to/offline.pt
names: [stream, offline]
input:
  manifest: /path/to/manifest.jsonl
  utt_id: "0"
streaming:
  # 覆盖时必须在 pt 的候选列表里，否则会报错（不做“兼容”）
  chunk_size: 64
  left_context_frames: 256
  tail_pad_frames: 30
```

```bash
python3 tools/force_align.py --config cfg.yaml --with-end
```

### 常见问题

- **为什么默认不是 `--text-mode auto`？**
  - 许多模型（尤其是英文大小写敏感的 BPE）对大小写有区分，强行改大小写可能改变对齐的语义；因此默认 `--text-mode raw`（文本“原样进入”）。
- **什么时候用 `auto`？**
  - 当原始文本的 `<unk>` 很多时，可以：
    - 显式使用 `--text-mode auto`（总是从 raw/upper/lower 中选 `<unk>` 最少的版本）
    - 或者使用 `--auto-text-if-unk-ge N`：仅当 raw 的 `<unk>` 数量达到阈值时才自动切到 `auto`
- **流式模型 chunk 参数如何确定？**
  - 默认从 `.pt` 的 `chunk_size/left_context_frames` 列表中选择；也可以用 `--chunk-size/--left-context-frames` 或 YAML 覆盖，但必须属于 `.pt` 列表（否则直接报错）。
