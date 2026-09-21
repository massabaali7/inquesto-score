#!/bin/bash
# Launch the v0.1 agent table: one Slurm job per configuration (resumable; re-run to continue).
#   bash examples/slurm/protocol_launch.sh            # submit all
#   bash examples/slurm/protocol_launch.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/../.."
CONFIGS=(
  "qwen3b-ep700|model=qwen2.5:3b endpointing_ms=700"
  "qwen7b-ep400|model=qwen2.5:7b endpointing_ms=400"
  "qwen7b-ep700|model=qwen2.5:7b endpointing_ms=700"
  "qwen7b-ep1100|model=qwen2.5:7b endpointing_ms=1100"
  "qwen14b-ep700|model=qwen2.5:14b endpointing_ms=700"
  "llama8b-ep700|model=llama3.1:8b endpointing_ms=700"
  "qwen32b-ep700|model=qwen2.5:32b endpointing_ms=700"
)
for c in "${CONFIGS[@]}"; do
  name=${c%%|*}; sets=${c#*|}
  if [ "$(ls runs/$name/calls 2>/dev/null | wc -l)" -ge 306 ]; then
    echo "skip $name (complete)"; continue
  fi
  if [ -n "${ONLY_IDLE:-}" ] && squeue -u "$USER" -h -o "%j" | grep -q "^iep-$name$"; then
    echo "skip $name (job running)"; continue
  fi
  if [ "${1:-}" = "--dry-run" ]; then echo "would submit $name: $sets"; continue; fi
  # shellcheck disable=SC2086
  jid=$(INQUESTO_SET="$sets" OUT="runs/$name" sbatch --parsable --job-name="iep-$name" ${SBATCH_ARGS:-} examples/slurm/protocol_run.sbatch)
  echo "submitted $name -> job $jid"
done
