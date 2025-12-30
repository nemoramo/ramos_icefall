# ramos_icefall

This repository is a working fork of `k2-fsa/icefall` (ASR recipes built on `k2` + `lhotse`).

- Legacy project overview: `README.legacy.md`
- Recipes live under `egs/<dataset>/<task>/...` (each recipe has its own README/RESULTS where applicable)

## Tools

- Force-alignment CLI: `tools/force_align.py` (usage + design notes: `tools/README.md`)

## Validation with WER (during training)

Some training scripts can compute and log **WER/CER during validation** (in addition to loss) using
**greedy decoding**.

Currently supported in this fork:

- `egs/librispeech/ASR/zipformer/train.py`: **word-level WER**.
- `egs/multi_zh-hans/ASR/zipformer/train.py`: **char-level CER**.

### How to enable

Add the following flags to your training command:

- `--compute-valid-wer true` to enable WER/CER computation.
- `--valid-interval 2000` to run validation every N training batches (**note:** it will also run at batch `0`).

Decoding type is inferred from the model setup:

- Transducer (`--use-transducer true`): RNNT greedy decoding.
- CTC (`--use-ctc true`): CTC greedy decoding.

### Example (Zipformer Transducer)

```bash
cd egs/librispeech/ASR
python3 zipformer/train.py \
  --exp-dir /path/to/exp \
  --manifest-dir /path/to/manifest_dir \
  --bpe-model /path/to/bpe.model \
  --use-transducer true \
  --compute-valid-wer true \
  --valid-interval 2000
```

Expected log line format:

```text
[valid] %WER 12.34% [123 / 999, 10 ins, 20 del, 93 sub ]
```

If TensorBoard is enabled (`--tensorboard true`), WER is also logged as:

- `train/valid_wer`

For `multi_zh-hans`, validation computes **CER** and logs:

- `train/valid_cer`

### Quick log filtering

```bash
rg "\\[valid\\] %(WER|CER)" /path/to/exp/log/log-train-*
```

## Audio notes (on-the-fly features)

For datasets with mixed encodings/sampling rates or slightly inaccurate `duration` metadata:

- Prefer torchaudio backend: `export LHOTSE_AUDIO_BACKEND=TorchaudioDefaultBackend`
- Resample when using on-the-fly fbank: `--on-the-fly-feats true --resample-to 16000`
- If you see `AudioLoadingError` about declared/loaded samples mismatch, increase tolerance (seconds), e.g.:
  `export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.2` (or larger for noisy manifests, e.g. `60`)
