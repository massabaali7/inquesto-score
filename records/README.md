# Inquesto v0.1 records

One numeric record per agent configuration (no audio, no transcripts). Regenerate a citation line with `python -c 'import json,sys; print(json.load(open(sys.argv[1]))["citation"])' records/<file>`.

| Agent (LLM / endpointing) | Citation line |
|---|---|
| gemini/gemini-2.5-flash / 700 ms | Inquesto v0.1: 22 ± 5 (B 17 · R 24 · I 22 · F 17), n = 306 |
| gpt-4.1-mini / 700 ms | Inquesto v0.1: 14 ± 4 (B 12 · R 13 · I 33 · F 11), n = 306 |
| gpt-5.4-mini / 700 ms | Inquesto v0.1: 6 ± 3 (B 4 · R 7 · I 11 · F 5), n = 306 |
| llama3.1:8b / 700 ms | Inquesto v0.1: 22 ± 5 (B 33 · R 22 · I 33 · F 17), n = 306 |
| qwen2.5:14b / 700 ms | Inquesto v0.1: 23 ± 5 (B 17 · R 24 · I 28 · F 20), n = 306 |
| qwen2.5:32b / 700 ms | Inquesto v0.1: 18 ± 4 (B 17 · R 16 · I 11 · F 15), n = 306 |
| qwen2.5:3b / 700 ms | Inquesto v0.1: 10 ± 3 (B 12 · R 8 · I 11 · F 5), n = 306 |
| qwen2.5:7b / 1100 ms | Inquesto v0.1: 31 ± 5 (B 29 · R 32 · I 17 · F 27), n = 306 |
| qwen2.5:7b / 400 ms | Inquesto v0.1: 22 ± 5 (B 25 · R 21 · I 28 · F 16), n = 306 |
| qwen2.5:7b / 700 ms | Inquesto v0.1: 28 ± 5 (B 29 · R 28 · I 33 · F 24), n = 306 |
| us.anthropic.claude-haiku-4-5-20251001-v1:0 / 700 ms | Inquesto v0.1: 26 ± 5 (B 17 · R 28 · I 33 · F 24), n = 306 |
| us.anthropic.claude-sonnet-4-6 / 700 ms | Inquesto v0.1: 19 ± 4 (B 21 · R 17 · I 28 · F 12), n = 306 |
| us.meta.llama3-1-8b-instruct-v1:0 / 700 ms | Inquesto v0.1: 15 ± 4 (B 8 · R 17 · I 17 · F 13), n = 306 |
