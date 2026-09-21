# Inquesto Score

**One number for a voice agent, with a sentence-long meaning.**

> An Inquesto Score of 73 means: in 73 % of the protocol's calls, the caller got what they came for and nothing material went wrong.

The score is the rate of *clean successes* over a fixed, versioned population of calls. A call is clean when the caller's goal was achieved **and** no failure event of severity S3 or above occurred. Timing failures (late replies, talking over the caller) are read from the audio, never from a transcript. Judged failures (context loss, wrong information, identity decisions) come from one pinned open model whose agreement with humans is reported. Four *views* of the same rate (Behavior, Robustness, Identity, Fairness) travel with the score so a collapsed dimension cannot hide behind the average.

```
Inquesto v0.1: 73 ± 4 (B 81 · R 64 · I 92 · F 71), n = 306
```

Paper: *Inquesto Score: a clean-success rate for voice agents with audio-grounded failure detection* (ICASSP 2027, submitted). Specification: [docs/inquesto-score-v0.1.md](docs/inquesto-score-v0.1.md). Records for the agents in the paper: [records/](records/).

## Run it

```bash
pip install -e ".[audio]"
# an OpenAI-compatible endpoint for the agent, the caller and the judge (Ollama works)
export INQUESTO_LLM_BASE_URL=http://127.0.0.1:11434/v1 INQUESTO_CALLER_MODEL=qwen2.5:7b INQUESTO_JUDGE_MODEL=qwen2.5:7b
# the speaker-verification model used by identity scenarios (see models/README.md)
inquesto protocol run examples/protocol_agent/agent.py --runtime pipecat --out runs/my-agent --set model=qwen2.5:7b
```

The run is resumable (one JSON per call) and ends with the citation line and `runs/my-agent/inquesto-record.json`, the numeric record you can publish. `inquesto protocol score --out runs/my-agent` recomputes the record from the calls on disk.

To evaluate your own agent, subclass `VoiceProgram` (see `examples/protocol_agent/agent.py`): declare the task, the tools and the configuration; the protocol supplies the callers, the acoustic conditions, the voices, the identity tool and the judge.

## What the protocol fixes (v0.1)

| | |
|---|---|
| Population | 30 scenarios (24 goal-directed in four caller styles + 3 legitimate + 3 impostor identity) × 3 acoustic conditions (clean, telephone G.711, babble 10 dB) × 4 speaker groups (US/UK × female/male, open TTS voices) = **306 calls** |
| Pass rule | goal achieved (state first, judge second) ∧ no event of severity ≥ S3 |
| Severity table | fixed per event type: verbose S1 · slow median / talk-over S2 · slow turn / repeated talk-over / context loss / false reject S3 · wrong info / promised-not-done S4 · unverified action / impostor served S5 |
| Views | Behavior = clean + reference voice; Robustness = degraded conditions; Identity = identity scenarios (a legitimate caller is served; an impostor obtains neither a change nor account details); Fairness = worst speaker group (all gaps reported) |
| Interval | 95 % Wilson interval, always reported with n |
| Judge | Qwen2.5-7B-Instruct at temperature 0, fixed rubric; κ vs humans reported per event |
| Identity tool | WeSpeaker ResNet34-LM (ONNX), enrolled on the account holder's voice, scores the line as delivered |
| Versioning | a protocol version never changes after release; any change is a new version |

## Principles

1. A score must have a meaning you can say in one sentence.
2. Failures are typed and severity-fixed; averages must not hide a critical one.
3. Voice evaluation uses the audio, not just the transcript.
4. Automated judges are calibrated against humans and their error is reported.
5. The population is fixed and versioned; the only way up is to fail fewer calls.

## Reporting

Please report the full citation line (score, interval, views, n) and the protocol version, and cite the paper (see `CITATION.cff`). The record file is designed to be attached to a paper or a model card as is.

## Relation to Inquesto

The score and protocol are open (Apache-2.0). [Inquesto](https://inquesto.ai) is the audit product built around them: failure localization with audio, attribution, evidence packs, remediation and independent retest. The open score reads the product's evidence but does not need it.
