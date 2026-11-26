#!/bin/bash
# Pack the results to send back: for every HoST run, its settings, learning curves,
# the last 2 checkpoints, the exported policy, and all evaluation results.
#
#   bash training/g1_host/cluster/collect.sh            # -> $SCRATCH/host_results_<date>.tar.gz
#
# $SCRATCH is purged of files not used for 60 days: send or copy the tarball soon.
set -euo pipefail
RUN_DIR="${RUN_DIR:-$SCRATCH/humanoid_runs}"
OUT="$SCRATCH/host_results_$(date +%Y-%m-%d_%H%M).tar.gz"
cd "$RUN_DIR"
LIST="$(mktemp)"
for run in logs/rsl_rl/host_*/*/; do
  [ -d "$run" ] || continue
  ls -1 "$run"model_*.pt 2>/dev/null | sed -E 's#^(.*model_([0-9]+)\.pt)$#\2 \1#' | sort -n | tail -2 \
    | cut -d' ' -f2- >> "$LIST" || true
  for f in "$run"params "$run"policy.onnx "$run"policy_meta.json "$run"events.out.tfevents.*; do
    [ -e "$f" ] && echo "$f" >> "$LIST"
  done
done
[ -d logs/eval ] && echo logs/eval >> "$LIST"
[ -d logs/videos/host ] && echo logs/videos/host >> "$LIST"
tar -czf "$OUT" -T "$LIST"
rm -f "$LIST"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1)). To copy it to your computer:"
echo "  scp ${USER:-<you>}@killarney.alliancecan.ca:$OUT ."
