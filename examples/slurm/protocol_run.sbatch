#!/bin/bash
# One agent configuration through Inquesto Protocol v0.1 (306 calls, real audio) on one GPU node.
# Resumable: re-submitting with the same OUT continues where it stopped.
#
#   INQUESTO_SET="model=qwen2.5:7b endpointing_ms=700" OUT=runs/qwen7b-ep700 sbatch examples/slurm/protocol_run.sbatch
#   INQUESTO_RUNTIME=local ...                      # transcript-only baseline (no audio events)
#
#SBATCH --account=bijn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --job-name=inquesto-protocol
#SBATCH --output=logs/inquesto-protocol-%j.out
set -euo pipefail
SIF=${INQUESTO_SIF:-/sw/llmhub/llmflux/containers/2.0.0/llm_processor.sif}
VENV=${INQUESTO_VENV:-$HOME/vamos/.venv}
OUT=${OUT:?set OUT=runs/<name>}
RUNTIME=${INQUESTO_RUNTIME:-pipecat}
SETS=${INQUESTO_SET:-"model=qwen2.5:7b"}
LIMIT=${INQUESTO_LIMIT:-0}
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/.ollama/models}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p "$OLLAMA_MODELS" logs
PORT=11434
for _ in $(seq 1 50); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then exec 3>&-; PORT=$((PORT + 1)); else break; fi
done
export APPTAINERENV_OLLAMA_MODELS="$OLLAMA_MODELS" APPTAINERENV_OLLAMA_HOST="127.0.0.1:$PORT" APPTAINERENV_OLLAMA_KEEP_ALIVE="12h"
apptainer exec --nv --bind "$OLLAMA_MODELS" "$SIF" ollama serve > "logs/ollama-$SLURM_JOB_ID.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 2; done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "ollama did not start"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${INQUESTO_REPO:-$HOME/vamos}"
export INQUESTO_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export INQUESTO_CALLER_MODEL=${INQUESTO_CALLER_MODEL:-qwen2.5:7b}
export INQUESTO_JUDGE_MODEL=${INQUESTO_JUDGE_MODEL:-gemma2:9b}
export INQUESTO_TTS_THREADS=${INQUESTO_TTS_THREADS:-8}
# Hosted agent LLM (optional): keys live in a file only you can read, never in this script or the repo.
#   ~/.config/inquesto/keys.env   e.g.  OPENAI_API_KEY=...  GEMINI_API_KEY=...  MISTRAL_API_KEY=...
KEYS=${INQUESTO_KEYS:-$HOME/.config/inquesto/keys.env}
# shellcheck disable=SC1090
[ -f "$KEYS" ] && set -a && source "$KEYS" && set +a
AGENT_MODEL=$(echo "$SETS" | tr ' ' '\n' | grep '^model=' | cut -d= -f2 || true)
case "${INQUESTO_AGENT_VENDOR:-local}" in
  openai)   export INQUESTO_AGENT_BASE_URL="https://api.openai.com/v1"                              INQUESTO_AGENT_API_KEY="${OPENAI_API_KEY:?}";;
  gemini)   export INQUESTO_AGENT_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai" INQUESTO_AGENT_API_KEY="${GEMINI_API_KEY:?}";;
  mistral)  export INQUESTO_AGENT_BASE_URL="https://api.mistral.ai/v1"                              INQUESTO_AGENT_API_KEY="${MISTRAL_API_KEY:?}";;
  deepseek) export INQUESTO_AGENT_BASE_URL="https://api.deepseek.com/v1"                            INQUESTO_AGENT_API_KEY="${DEEPSEEK_API_KEY:?}";;
  xai)      export INQUESTO_AGENT_BASE_URL="https://api.x.ai/v1"                                    INQUESTO_AGENT_API_KEY="${XAI_API_KEY:?}";;
  litellm)  export INQUESTO_AGENT_BASE_URL="${LITELLM_BASE_URL:-https://ai-gateway.andrew.cmu.edu/v1}" INQUESTO_AGENT_API_KEY="${LITELLM_API_KEY:?}";;
  local)    ;;
  *) echo "unknown INQUESTO_AGENT_VENDOR=$INQUESTO_AGENT_VENDOR"; exit 1;;
esac
LOCAL_MODELS="$INQUESTO_CALLER_MODEL $INQUESTO_JUDGE_MODEL"
[ "${INQUESTO_AGENT_VENDOR:-local}" = "local" ] && LOCAL_MODELS="${AGENT_MODEL:-qwen2.5:7b} $LOCAL_MODELS"
for m in $(printf '%s\n' $LOCAL_MODELS | sort -u); do
  apptainer exec --bind "$OLLAMA_MODELS" "$SIF" ollama pull "$m" 2>&1 | tail -1
done
SET_ARGS=(); for kv in $SETS; do SET_ARGS+=(--set "$kv"); done
LIMIT_ARGS=(); [ "$LIMIT" != "0" ] && LIMIT_ARGS=(--limit "$LIMIT")
[ -n "${INQUESTO_ONLY:-}" ] && LIMIT_ARGS+=(--only "$INQUESTO_ONLY")
echo "== protocol v0.1  runtime=$RUNTIME  agent vendor=${INQUESTO_AGENT_VENDOR:-local}  sets=$SETS  out=$OUT  node=$(hostname -s)  $(date) =="
inquesto protocol run examples/protocol_agent/agent.py --runtime "$RUNTIME" --out "$OUT" "${SET_ARGS[@]}" "${LIMIT_ARGS[@]}" \
  2> >(grep -v -E "DEBUG|INFO \||DeprecationWarning|^\s+task = |^\s+runner = |Creating a resampler|in_sample_rate|output_sample_rate|^\s*$" >&2)
echo "== done $(date) =="
