# France ASR (Zipformer) Notes

This recipe is intended for large-scale French ASR training with **on-the-fly**
feature extraction (Lhotse `OnTheFlyFeatures` + Fbank).

When training at high concurrency (8 GPUs * many dataloader workers), audio I/O
from a shared NFS mount can become unstable or slow. The recommended approach is
to read audio from the TOS bucket `asr-audio-data` via an fsx/hpvs FUSE mount.

## Entry Points

- Training: `egs/france/ASR/run_french_150M.sh`
- Main trainer: `egs/france/ASR/zipformer/train.py`

## Data Layout (Expected)

`run_french_150M.sh` expects:

- `MANIFEST_DIR`: directory containing Lhotse cuts (JSONL.GZ):
  - `msr_cuts_French_train.jsonl.gz`
  - `msr_cuts_French_valid.jsonl.gz`
  - `msr_cuts_French_test.jsonl.gz`
- `BPE_MODEL`: a French SentencePiece model (default: `lang_bpe_2048/bpe.model`)

The cuts must reference audio paths that are readable from the training host.

## Use TOS Mount For Stable I/O

On-the-fly training reads and decodes audio in dataloader workers. To reduce NFS
contention and improve stability, mount the TOS bucket and make sure your cuts
use the mount prefix.

### Quick Checks

```bash
findmnt /mnt/asr-audio-data
grep -F " /mnt/asr-audio-data " /proc/mounts
```

### Mount (fsx/hpvs fusedaemon)

```bash
sudo /opt/fsx/tools/start_fusedaemon.sh \
  --fusedaemon_combination_name=tos \
  --fusedaemon_support_instance_list=tos

sudo mkdir -p /mnt/asr-audio-data
sudo mount -t fsx /tos/asr-audio-data /mnt/asr-audio-data \
  -o region="ap-southeast-1",endpoint="https://tos-ap-southeast-1.ivolces.com",credential_filepath="/opt/credential.json",no_writeback_cache

findmnt /mnt/asr-audio-data
```

Notes:
- Do **not** commit credentials. The mount reads AK/SK from a JSON file like
  `/opt/credential.json`.
- Mount endpoint uses `tos-<region>...`; S3-compatible API uses `tos-s3-<region>...`.

## Ensure Cuts Point To The Mount Prefix

If your cuts currently reference an NFS prefix, you can rewrite them:

```bash
python egs/france/ASR/local/replace_cut_source_prefix.py \
  --input-cuts  "${MANIFEST_DIR}/msr_cuts_French_train.jsonl.gz" \
  --output-cuts "${MANIFEST_DIR}/msr_cuts_French_train.tos.jsonl.gz" \
  --src-prefix  "/nfs/audio_root/" \
  --dst-prefix  "/mnt/asr-audio-data/audio_root/"
```

## Tools: Build Cuts Without `lhotse kaldi import`

For extremely large datasets where `lhotse kaldi import` can stall under heavy
I/O contention, a fast streaming converter is available:

`egs/france/ASR/local/json_to_lhotse_cuts_fast.py`

It converts a JSONL manifest containing `audio_filepath` + `duration` (+ text)
into Lhotse-style cuts JSONL.GZ.

## Benchmark: FUSE Mount vs Direct TOS API

To compare local mount reads vs direct TOS S3-compatible reads (boto3):

`egs/france/ASR/local/benchmark_french_tos_vs_fuse.py`

The script needs TOS config via environment variables or an optional `.env`
file:

- `TOS_ACCESS_KEY_ID`
- `TOS_SECRET_ACCESS_KEY`
- `TOS_ENDPOINT` (S3 endpoint, e.g. `https://tos-s3-<region>.ivolces.com`)
- `TOS_REGION`

