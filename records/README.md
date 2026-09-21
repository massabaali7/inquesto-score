# Inquesto v0.1 records

One numeric record per agent configuration (no audio, no transcripts). Regenerate a citation line with `python -c 'import json,sys; print(json.load(open(sys.argv[1]))["citation"])' records/<file>`.

| Agent (LLM / endpointing) | Citation line |
|---|---|
| llama3.1:8b / 700 ms | Inquesto v0.1: 22 ± 5 (B 33 · R 22 · I 33 · F 17), n = 306 |
| qwen2.5:14b / 700 ms | Inquesto v0.1: 23 ± 5 (B 17 · R 24 · I 28 · F 20), n = 306 |
| qwen2.5:32b / 700 ms | Inquesto v0.1: 18 ± 4 (B 17 · R 16 · I 11 · F 15), n = 306 |
| qwen2.5:3b / 700 ms | Inquesto v0.1: 10 ± 3 (B 12 · R 8 · I 11 · F 5), n = 306 |
| qwen2.5:7b / 1100 ms | Inquesto v0.1: 31 ± 5 (B 29 · R 32 · I 17 · F 27), n = 306 |
| qwen2.5:7b / 400 ms | Inquesto v0.1: 22 ± 5 (B 25 · R 21 · I 28 · F 16), n = 306 |
| qwen2.5:7b / 700 ms | Inquesto v0.1: 28 ± 5 (B 29 · R 28 · I 33 · F 24), n = 306 |
