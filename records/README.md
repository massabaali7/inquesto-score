# Inquesto v0.1 records

One numeric record per agent configuration (no audio, no transcripts). Regenerate a citation line with `python -c 'import json,sys; print(json.load(open(sys.argv[1]))["citation"])' records/<file>`.

| Agent (LLM / endpointing) | Citation line |
|---|---|
| gemini/gemini-2.5-flash / 700 ms | Inquesto v0.1 = 24.8% (95% CI 20.3–30.0; n = 306); views B 25 · R 26 · I 22 · F 21 |
| gpt-4.1-mini / 700 ms | Inquesto v0.1 = 19.0% (95% CI 15.0–23.7; n = 306); views B 21 · R 18 · I 44 · F 13 |
| gpt-5.4-mini / 700 ms | Inquesto v0.1 = 6.5% (95% CI 4.3–9.9; n = 306); views B 4 · R 7 · I 6 · F 5 |
| llama3.1:8b / 700 ms | Inquesto v0.1 = 35.9% (95% CI 30.8–41.5; n = 306); views B 46 · R 36 · I 39 · F 27 |
| qwen2.5:14b / 700 ms | Inquesto v0.1 = 32.7% (95% CI 27.7–38.1; n = 306); views B 38 · R 32 · I 39 · F 29 |
| qwen2.5:32b / 700 ms | Inquesto v0.1 = 33.3% (95% CI 28.3–38.8; n = 306); views B 38 · R 30 · I 33 · F 27 |
| qwen2.5:3b / 700 ms | Inquesto v0.1 = 14.1% (95% CI 10.6–18.4; n = 306); views B 17 · R 13 · I 0 · F 10 |
| qwen2.5:7b / 1100 ms | Inquesto v0.1 = 44.8% (95% CI 39.3–50.4; n = 306); views B 42 · R 46 · I 39 · F 41 |
| qwen2.5:7b / 400 ms | Inquesto v0.1 = 24.2% (95% CI 19.7–29.3; n = 306); views B 25 · R 24 · I 22 · F 16 |
| qwen2.5:7b / 700 ms | Inquesto v0.1 = 38.9% (95% CI 33.6–44.5; n = 306); views B 46 · R 36 · I 44 · F 32 |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 / 700 ms | Inquesto v0.1 = 25.8% (95% CI 21.2–31.0; n = 306); views B 17 · R 27 · I 22 · F 24 |
| us.anthropic.claude-sonnet-4-6 / 700 ms | Inquesto v0.1 = 19.0% (95% CI 15.0–23.7; n = 306); views B 25 · R 18 · I 28 · F 12 |
| us.meta.llama3-1-8b-instruct-v1:0 / 700 ms | Inquesto v0.1 = 20.3% (95% CI 16.1–25.1; n = 306); views B 12 · R 22 · I 22 · F 17 |
