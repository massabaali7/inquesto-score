# Support agent example

The reference agent behind the Inquesto demo and the `support-v1` benchmark.

```bash
inquesto evaluate examples/support_agent/agent.py --failures
inquesto optimize examples/support_agent/agent.py --constraint cost<=0.05 --constraint latency<=950 -v
inquesto evaluate examples/support_agent/agent.py --save-baseline .inquesto/baseline.json
inquesto gate examples/support_agent/agent.py --baseline .inquesto/baseline.json
```

Expected on the mock runtime: 78% task success at baseline, 90% after the
constrained search (endpointing 700 -> 900 ms, context 8 -> 12 turns,
temperature 0.7 -> 0.2), gate clear. `tests/test_loop.py` pins these numbers.

`support-v1` is 40 billing calls across five caller styles (neutral, fast,
hesitant, accented, noisy). Roughly a tenth of them are deliberately
unsolvable — calls that need a human. An optimizer that reports 100% on this
suite has a bug.

Everything here runs on the deterministic `mock` runtime, so it works with no
API keys. Swap `--runtime` once a real adapter lands.
