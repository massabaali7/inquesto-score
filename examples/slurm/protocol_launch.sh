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
# Hosted LLMs as the agent (caller, judge, STT, TTS unchanged). Only when the keys file exists.
KEYS=${INQUESTO_KEYS:-$HOME/.config/inquesto/keys.env}
if [ -f "$KEYS" ]; then
  # shellcheck disable=SC1090
  set -a; source "$KEYS"; set +a
  [ -n "${OPENAI_API_KEY:-}" ]   && CONFIGS+=("gpt4omini-ep700|model=gpt-4o-mini endpointing_ms=700|openai" "gpt41-ep700|model=gpt-4.1 endpointing_ms=700|openai")
  [ -n "${GEMINI_API_KEY:-}" ]   && CONFIGS+=("gemini25flash-ep700|model=gemini-2.5-flash endpointing_ms=700|gemini")
  [ -n "${MISTRAL_API_KEY:-}" ]  && CONFIGS+=("mistralsmall-ep700|model=mistral-small-latest endpointing_ms=700|mistral")
  [ -n "${DEEPSEEK_API_KEY:-}" ] && CONFIGS+=("deepseekchat-ep700|model=deepseek-chat endpointing_ms=700|deepseek")
  [ -n "${XAI_API_KEY:-}" ]      && CONFIGS+=("grok3mini-ep700|model=grok-3-mini endpointing_ms=700|xai")
  # One gateway, many models (LiteLLM, e.g. the CMU AI gateway): INQUESTO_LITELLM_MODELS="gpt-4o-mini,claude-sonnet-4,gemini-2.5-flash"
  if [ -n "${LITELLM_API_KEY:-}" ] && [ -n "${INQUESTO_LITELLM_MODELS:-}" ]; then
    for m in ${INQUESTO_LITELLM_MODELS//,/ }; do
      n=$(echo "$m" | tr -c 'a-zA-Z0-9\n' '-' | sed 's/-*$//')
      CONFIGS+=("$n-ep700|model=$m endpointing_ms=700|litellm")
    done
  fi
fi
for c in "${CONFIGS[@]}"; do
  name=${c%%|*}; rest=${c#*|}; sets=${rest%%|*}; vendor=local; [[ "$rest" == *"|"* ]] && vendor=${rest##*|}
  if [ "$(ls runs/$name/calls 2>/dev/null | wc -l)" -ge 306 ]; then
    echo "skip $name (complete)"; continue
  fi
  if [ -n "${ONLY_IDLE:-}" ] && squeue -u "$USER" -h -o "%j" | grep -q "^iep-$name$"; then
    echo "skip $name (job running)"; continue
  fi
  if [ "${1:-}" = "--dry-run" ]; then echo "would submit $name: $sets"; continue; fi
  # shellcheck disable=SC2086
  jid=$(INQUESTO_AGENT_VENDOR="$vendor" INQUESTO_SET="$sets" OUT="runs/$name" sbatch --parsable --job-name="iep-$name" ${SBATCH_ARGS:-} examples/slurm/protocol_run.sbatch)
  echo "submitted $name -> job $jid"
done
