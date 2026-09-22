# Inquesto v0.1 records

One numeric record per agent configuration (no audio, no transcripts). Regenerate a citation line with `python -c 'import json,sys; print(json.load(open(sys.argv[1]))["citation"])' records/<file>`.

| Agent (LLM / endpointing) | Citation line |
|---|---|
| gemini/gemini-2.5-flash / 700 ms | Inquesto v0.1 = 22.2% (95% CI 17.9–27.2; n = 306); views B 17 · R 24 · I 22 · F 17 |
| gpt-4.1-mini / 700 ms | Inquesto v0.1 = 14.1% (95% CI 10.6–18.4; n = 306); views B 12 · R 13 · I 33 · F 11 |
| gpt-5.4-mini / 700 ms | Inquesto v0.1 = 6.2% (95% CI 4.0–9.5; n = 306); views B 4 · R 7 · I 11 · F 5 |
| llama3.1:8b / 700 ms | Inquesto v0.1 = 21.6% (95% CI 17.3–26.5; n = 306); views B 33 · R 22 · I 33 · F 17 |
| qwen2.5:14b / 700 ms | Inquesto v0.1 = 23.2% (95% CI 18.8–28.2; n = 306); views B 17 · R 24 · I 28 · F 20 |
| qwen2.5:32b / 700 ms | Inquesto v0.1 = 18.0% (95% CI 14.1–22.7; n = 306); views B 17 · R 16 · I 11 · F 15 |
| qwen2.5:3b / 700 ms | Inquesto v0.1 = 9.8% (95% CI 7.0–13.7; n = 306); views B 12 · R 8 · I 11 · F 5 |
| qwen2.5:7b / 1100 ms | Inquesto v0.1 = 31.0% (95% CI 26.1–36.4; n = 306); views B 29 · R 32 · I 17 · F 27 |
| qwen2.5:7b / 400 ms | Inquesto v0.1 = 21.6% (95% CI 17.3–26.5; n = 306); views B 25 · R 21 · I 28 · F 16 |
| qwen2.5:7b / 700 ms | Inquesto v0.1 = 28.4% (95% CI 23.7–33.7; n = 306); views B 29 · R 28 · I 33 · F 24 |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 / 700 ms | Inquesto v0.1 = 26.1% (95% CI 21.5–31.3; n = 306); views B 17 · R 28 · I 33 · F 24 |
| us.anthropic.claude-sonnet-4-6 / 700 ms | Inquesto v0.1 = 18.6% (95% CI 14.7–23.4; n = 306); views B 21 · R 17 · I 28 · F 12 |
| us.meta.llama3-1-8b-instruct-v1:0 / 700 ms | Inquesto v0.1 = 14.7% (95% CI 11.2–19.1; n = 306); views B 8 · R 17 · I 17 · F 13 |
