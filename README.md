<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/inquesto-wordmark-dark.svg">
    <img src="assets/inquesto-wordmark-light.svg" alt="Inquesto" width="360">
  </picture>
</p>

<p align="center"><b>One number for a voice agent, with a sentence-long meaning.</b></p>

<p align="center">
<code>Inquesto v0.1: 31 ± 5 (B 29 · R 32 · I 17 · F 27), n = 306</code>
</p>

> **An Inquesto Score of 73 means: in 73 % of the protocol's calls, the caller got what they came for and nothing material went wrong.**

The Inquesto Score is the rate of *clean successes* over a fixed, versioned population of 306 calls with real audio. A call is clean when the caller's goal was achieved **and** no failure event of severity S3 or above occurred. Timing failures (late replies, talking over the caller) are read from the audio, never from a transcript. Judged failures (context loss, wrong information, identity decisions) come from one pinned open model. Four *views* of the same rate travel with the score: **B**ehavior, **R**obustness, **I**dentity, **F**airness.

Paper: *Inquesto Score: a clean-success rate for voice agents with audio-grounded failure detection* (Baali & Raj, ICASSP 2027, submitted). Specification: [docs/inquesto-score-v0.1.md](docs/inquesto-score-v0.1.md). Records for the 13 agents in the paper: [records/](records/).

## Get a score in three commands

```bash
pip install "inquesto-score[audio]"      # Python 3.10+
inquesto setup                            # downloads the verifier, checks your LLM endpoint
inquesto score my_agent.py                # 306 calls later: the citation line + inquesto-runs/<agent>/inquesto-record.json
```

`inquesto setup` needs an OpenAI-compatible endpoint that serves the protocol's pinned caller and judge model, Qwen2.5-7B-Instruct. The easiest is [Ollama](https://ollama.com): `ollama pull qwen2.5:7b && ollama serve` (default `INQUESTO_LLM_BASE_URL=http://127.0.0.1:11434/v1`). Kokoro (TTS) and Whisper (STT) download themselves on first use. A GPU makes the caller and judge fast; the audio stack runs on CPU. A full run is about 2 hours on one GPU node; `--limit 20` gives a 10-minute smoke test.

## Define your agent

Your agent is a `VoiceProgram`: what it is for, which tools it has, and the model that runs it. The protocol supplies everything else: the callers, the acoustic conditions, the voices, the verifier, the judge.

```python
# my_agent.py
from inquesto import VoiceProgram

class BillingAgent(VoiceProgram):
    task = ("You are the billing support agent for Acme. Look the account up before discussing it. "
            "Verify the caller's voice before any account change. Keep replies to two short sentences.")
    tools = ["lookup_account", "verify_voice", "issue_refund", "update_card", "update_email", "cancel_subscription"]
```

```bash
inquesto score my_agent.py --set model=qwen2.5:7b --set endpointing_ms=700
```

The model behind the agent can live anywhere with an OpenAI-compatible API. To score a hosted model while the caller and judge stay pinned locally:

```bash
export INQUESTO_AGENT_BASE_URL=https://api.openai.com/v1 INQUESTO_AGENT_API_KEY=$OPENAI_API_KEY
inquesto score my_agent.py --set model=gpt-4.1-mini
```

Or from Python:

```python
from inquesto import score
rec = score(BillingAgent, model="gpt-4.1-mini")
print(rec["citation"])          # Inquesto v0.1: 14 ± 4 (B 12 · R 13 · I 33 · F 11), n = 306
print(rec["views"]["fairness"]) # per speaker group, with each group's gap to the reference voice
```

## Bring your own pipeline

If your agent is not "a prompt plus tools behind an LLM" but a running voice system, implement the runtime interface and the protocol drives it the same way:

```python
from inquesto.adapters import register
from inquesto.program import Conversation

@register("mine")
class MyRuntime:
    name = "mine"
    def run(self, program, scenario, seed) -> Conversation:
        ...  # play the caller's audio into your system, record what it says and when, return the turns
```

`inquesto score my_agent.py --runtime mine`. A `Conversation` carries the turns with start and end times, the tool calls, the response latencies and whether a turn started while the caller was still speaking; the detectors and the judge do the rest. An audio-over-WebSocket bridge for arbitrary agents is planned for v0.2.

## What the protocol fixes (v0.1)

| | |
|---|---|
| Population | 30 scenarios (24 goal-directed in four caller styles + 3 legitimate + 3 impostor identity) × 3 acoustic conditions (clean, telephone G.711, babble 10 dB) × 4 speaker groups (US/UK × female/male, open TTS voices) = **306 calls** |
| Pass rule | goal achieved (state first, judge second) ∧ no event of severity ≥ S3 |
| Severity table | fixed per event type: verbose S1 · slow median / talk-over S2 · slow turn / repeated talk-over / context loss / false reject S3 · wrong info / promised-not-done S4 · unverified action / impostor served S5 |
| Views | Behavior = clean audio, reference voice; Robustness = degraded conditions; Identity = identity scenarios; Fairness = worst speaker group, every gap reported |
| Interval | 95 % Wilson interval, always reported with n |
| Judge | Qwen2.5-7B-Instruct at temperature 0, fixed rubric; agreement with humans reported per event |
| Identity tool | WeSpeaker ResNet34-LM (ONNX) enrolled on the account holder's voice, threshold calibrated so the verifier itself makes no error on the population |
| Versioning | a protocol version never changes after release; any change is a new version |

## Results in the paper

Thirteen configurations of one reference agent, 3,978 calls. Full records with views, failure counts by severity and type, and the transcript-only counterfactual are in [records/](records/).

| Agent (LLM / endpointing) | Inquesto v0.1 |
|---|---|
| Qwen2.5-7B / 1100 ms | 31 ± 5 (B 29 · R 32 · I 17 · F 27) |
| Qwen2.5-7B / 700 ms | 28 ± 5 (B 29 · R 28 · I 33 · F 24) |
| Claude Haiku 4.5, hosted | 26 ± 5 (B 17 · R 28 · I 33 · F 24) |
| Qwen2.5-14B / 700 ms | 23 ± 5 (B 17 · R 24 · I 28 · F 20) |
| Gemini 2.5 Flash, hosted | 22 ± 5 (B 17 · R 24 · I 22 · F 17) |
| Qwen2.5-7B / 400 ms | 22 ± 5 (B 25 · R 21 · I 28 · F 16) |
| Llama-3.1-8B / 700 ms | 22 ± 5 (B 33 · R 22 · I 33 · F 17) |
| Claude Sonnet 4.6, hosted | 19 ± 4 (B 21 · R 17 · I 28 · F 12) |
| Qwen2.5-32B / 700 ms | 18 ± 4 (B 17 · R 16 · I 11 · F 15) |
| Llama-3.1-8B, hosted | 15 ± 4 (B 8 · R 17 · I 17 · F 13) |
| GPT-4.1-mini, hosted | 14 ± 4 (B 12 · R 13 · I 33 · F 11) |
| Qwen2.5-3B / 700 ms | 10 ± 3 (B 12 · R 8 · I 11 · F 5) |
| GPT-5.4-mini, hosted | 6 ± 3 (B 4 · R 7 · I 11 · F 5) |

Absolute scores are low on purpose: the reference agents are small models behind an open STT/VAD stack, and hosted models were reached through a gateway that adds a second of latency to every reply. The protocol compares configurations of one pipeline on equal terms; it does not flatter anyone.

## Principles

1. A score must have a meaning you can say in one sentence.
2. Failures are typed and severity-fixed; averages must not hide a critical one.
3. Voice evaluation uses the audio, not just the transcript.
4. Automated judges are calibrated against humans and their error is reported.
5. The population is fixed and versioned; the only way up is to fail fewer calls.

## Reporting a score

Report the full citation line (score, interval, views, n) and the protocol version, attach `inquesto-record.json` to the paper or model card, and cite the paper (`CITATION.cff`). Re-scoring a finished directory after a detector change: `inquesto protocol score --out inquesto-runs/<agent>`.

## Relation to Inquesto

The score and the protocol are open (Apache-2.0). [Inquesto](https://inquesto.ai) is the audit product built around them: failure localization with audio, attribution, evidence packs, remediation and independent retest. The open score reads the product's evidence but does not need it. Try the audit demo at [inquesto.ai/demo](https://inquesto.ai/demo).
