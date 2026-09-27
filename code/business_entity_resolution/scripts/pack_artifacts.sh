#!/usr/bin/env bash
# On the development machine: bundle the artefacts run_experiments.sh needs (no retraining of the
# bi-encoder / blocking / stage 1 / existing cross-encoders on the DGX). ~24 GB, parquet is
# already compressed so the tar is uncompressed.
# Usage: bash scripts/pack_artifacts.sh --list > files.txt   then rsync -avP --files-from=files.txt . host:repo/
#        bash scripts/pack_artifacts.sh /path/to/ber_artifacts.tar      (needs ~24 GB free there)
#        bash scripts/pack_artifacts.sh - | ssh dgx 'cd /path/repo && tar -xf -'   (stream, no local copy)
#   then on the DGX, from the repository root:  tar -xf ber_artifacts.tar
source "$(dirname "$0")/common.sh"
OUT=${1:?usage: pack_artifacts.sh <out.tar | ->}
cd "$BER_ROOT"
files=()
for s in train test; do
  for f in source1.parquet source2.parquet source3.parquet candidates.parquet p1.parquet \
           ce.parquet ce_b.parquet ce_c.parquet ce_l.parquet; do
    [ -f "work/$s/$f" ] && files+=("work/$s/$f")
  done
done
files+=(work/train/gt.parquet work/train/subset_s1.npy)
for f in work/test/p2_v6.parquet work/test/p2_v8.parquet work/models/decision_v6.json work/models/decision_v8.json; do
  [ -f "$f" ] && files+=("$f")
done
if [ "$OUT" = "--list" ]; then  # file list for rsync --files-from (resumable transfer, no tar)
  printf '%s\n' "${files[@]}"
  exit 0
fi
printf '%s\n' "${files[@]}" >&2
if [ "$OUT" = "-" ]; then
  tar -cf - "${files[@]}"
else
  tar -cf "$OUT" "${files[@]}"
  ls -la "$OUT" >&2
fi
