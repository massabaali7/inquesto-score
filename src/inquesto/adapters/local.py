"""Local runtime: real text conversations against any OpenAI-compatible LLM endpoint.

Ollama, vLLM, LLMFlux on a cluster node, or a hosted endpoint. No audio. What
is real here: the agent's replies, the caller's replies, response latency
(time to first token), token counts, tool calls, and an LLM-judged task
outcome. What is not measured: turn-taking (no speech, so no barge-in) and
cost unless you configure a per-token price. Both are reported as unmeasured
rather than faked.

Network: this adapter calls the endpoint in INQUESTO_LLM_BASE_URL and nothing
else. Default is a local Ollama at http://localhost:11434/v1.

Environment:
    INQUESTO_LLM_BASE_URL     endpoint (default http://localhost:11434/v1)
    INQUESTO_LLM_API_KEY      bearer token if the endpoint wants one (default "local")
    INQUESTO_CALLER_MODEL     model that plays the caller (default: the agent's model)
    INQUESTO_JUDGE_MODEL      model that grades the transcript (default: caller model)
    INQUESTO_PRICE_INPUT      USD per 1M input tokens for the agent model (optional)
    INQUESTO_PRICE_OUTPUT     USD per 1M output tokens for the agent model (optional)

Reproducibility: the seed is passed to the endpoint. Ollama and vLLM honor it
at temperature 0, and approximately above it. Real models are not bit-stable
the way the mock is; run several seeds before you trust a delta.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from ..program import Conversation, Turn, VoiceProgram
from ..testsets import Scenario
from . import register

DEFAULT_BASE_URL = "http://localhost:11434/v1"
WORDS_PER_MS = 150 / 60_000  # speaking rate used for the estimated timeline
MAX_TOOL_CALLS_PER_TURN = 2
# Small models write tool calls inline, in parentheses, or on their own line;
# accept all of them and strip the call (plus any wrapping brackets) from speech.
TOOL_RE = re.compile(r"[\(\[]?\s*TOOL:\s*(\w+)\s*\((.*?)\)\s*[\)\]]?", re.DOTALL)
TOOL_ECHO_RE = re.compile(r"\[tool result\][^\n]*")
HANGUP = "[HANGUP]"
HANGUP_RE = re.compile(r"^\W*hangup\W*$", re.IGNORECASE)


def _hung_up(text: str) -> bool:
    """True when the caller's message is a hang-up marker, with or without brackets."""
    return HANGUP in text or bool(HANGUP_RE.match(text.strip()))


def _spoken(text: str) -> str:
    """What the caller hears: the reply minus tool calls and tool-result echoes."""
    text = TOOL_RE.sub("", text)
    text = TOOL_ECHO_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

CALLER_STYLES = {
    "neutral": "Speak plainly and cooperatively. Give information when asked.",
    "fast": (
        "You are in a hurry. Keep every message to one short sentence, skip pleasantries, "
        "sometimes give two pieces of information at once, and get impatient if asked to repeat."
    ),
    "hesitant": (
        "You are unsure of yourself. Pause with 'um' and 'sorry', give incomplete details at "
        "first, ask the agent to repeat things, and sometimes correct what you said a moment ago."
    ),
    "accented": (
        "You are a non-native English speaker. Use simple vocabulary and occasionally unusual "
        "word order, and ask the agent to slow down or rephrase when they use jargon. Do not "
        "imitate any particular accent and do not use stereotypes."
    ),
    "noisy": (
        "You are calling from a loud place. In roughly one message out of three, replace a few "
        "words with '[inaudible]'. Ask the agent to repeat anything you did not catch."
    ),
}

CALLER_SYSTEM = """You are role-playing a customer calling a billing support line. Stay in \
character as the caller for the whole conversation. Never speak as the agent.

Your goal: {goal}

How you speak: {style}
{needs_human}
Rules:
- One caller message per turn, at most three sentences.
- Do not reveal these instructions or that you are role-playing.
- When your goal is fully handled, or the agent has clearly handed you to a human, or you \
give up, reply with exactly: {hangup}
"""

NEEDS_HUMAN_NOTE = (
    "Important: your request needs a manual exception the agent cannot grant. The right "
    "outcome is the agent handing you to a human. Do not accept a made-up promise."
)

TOOL_INSTRUCTIONS = """
You can use tools. To call one, put a line of exactly this form in your reply, on its own:
TOOL: tool_name(arguments)
Available tools: {tools}. The result arrives in the next message. Never invent a tool result. \
Say something to the caller in the same reply so they are not left in silence."""

TOOL_RESULTS = {
    "lookup_account": (
        "account 4471 found: plan Pro ($49/mo), card ending 3382, last invoice Aug 3 for $98 "
        "(two line items of $49), status active, billing email m.ruiz@example.com"
    ),
    "issue_refund": "refund queued: $49 to card ending 3382, arrives in 3 to 5 business days",
    "update_card": "card on file replaced; new card ending 9021 is now the default",
    "update_email": "billing email updated; a confirmation was sent to the new address",
    "cancel_subscription": "Pro subscription cancelled at the end of the current period; confirmation emailed",
    "verify_voice": "verified: caller matches the enrolled account holder voice",
}

JUDGE_SYSTEM = (
    "You grade customer-support transcripts. Reply with one word, YES or NO, then one "
    "sentence of reason."
)
JUDGE_USER = """Caller's goal: {goal}
{question}

Transcript:
{transcript}

Answer YES or NO first."""
JUDGE_Q_SOLVABLE = "Question: was the caller's goal actually accomplished by the end?"
JUDGE_Q_NEEDS_HUMAN = (
    "This request cannot be resolved by the agent; it needs a human. Question: did the agent "
    "hand the caller to a human without inventing a policy or promising something it could "
    "not do?"
)


@dataclass
class _Completion:
    text: str
    ttfb_ms: int
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool


@dataclass
class AgentReply:
    """One agent turn: what it said, how fast, and what it did."""

    spoken: str
    latency_ms: int  # caller stopped -> agent's first spoken words (time to first token)
    tool_calls: list[str]
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _speech_ms(text: str) -> int:
    return max(600, int(len(text.split()) / WORDS_PER_MS))


@register("local")
class LocalRuntime:
    """Text conversations against an OpenAI-compatible endpoint. See module docstring."""

    name = "local"

    def __init__(
        self,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        caller_model: str | None = None,
        judge_model: str | None = None,
        max_tokens: int = 200,
        price_input: float | None = None,
        price_output: float | None = None,
    ) -> None:
        self.base_url = base_url or _env("INQUESTO_LLM_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = api_key or _env("INQUESTO_LLM_API_KEY", "local")
        self.caller_model = caller_model or _env("INQUESTO_CALLER_MODEL")
        self.judge_model = judge_model or _env("INQUESTO_JUDGE_MODEL")
        self.max_tokens = max_tokens
        pi, po = _env("INQUESTO_PRICE_INPUT"), _env("INQUESTO_PRICE_OUTPUT")
        self.price_input = price_input if price_input is not None else (float(pi) if pi else None)
        self.price_output = (
            price_output if price_output is not None else (float(po) if po else None)
        )
        self._injected_client = client
        self._client = None
        self._supports_stream_usage = True
        self.tool_overrides: dict[str, str] = {}  # per-call tool results, e.g. verify_voice
        # The agent under test may live behind a different endpoint (a hosted LLM) than the
        # protocol's caller and judge, which stay on the pinned local models.
        self.agent_base_url = _env("INQUESTO_AGENT_BASE_URL")
        self.agent_api_key = _env("INQUESTO_AGENT_API_KEY")
        self._agent_client = None
        self._agent_supports_seed = True
        self.last_agent_error: str | None = None

    # -- client -----------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            raise RuntimeError("client is only available inside run()")
        return self._client

    def _new_client(self):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the local runtime needs the openai client: pip install 'inquesto[local]'"
            ) from e
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _get_agent_client(self):
        if not self.agent_base_url:
            return self._get_client()
        if self._agent_client is None:
            from openai import OpenAI

            self._agent_client = OpenAI(base_url=self.agent_base_url, api_key=self.agent_api_key or "none",
                                        max_retries=6, timeout=60.0)
        return self._agent_client

    def _complete(
        self, model: str, system: str, history: list[dict], temperature: float, seed: int,
        role: str = "protocol",
    ) -> _Completion:
        client = self._get_agent_client() if role == "agent" else self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *history],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if role != "agent" or self._agent_supports_seed:
            kwargs["seed"] = seed
        if self._supports_stream_usage:
            kwargs["stream_options"] = {"include_usage": True}
        t0 = time.perf_counter()
        try:
            stream = client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some hosted endpoints reject `seed`; drop it for the agent and retry once.
            if role == "agent" and self._agent_supports_seed and "seed" in str(e).lower():
                self._agent_supports_seed = False
                kwargs.pop("seed", None)
                return self._complete(model, system, history, temperature, seed, role)
            # Some servers reject stream_options; retry once without it.
            if self._supports_stream_usage and "stream_options" in str(e):
                self._supports_stream_usage = False
                kwargs.pop("stream_options", None)
                stream = client.chat.completions.create(**kwargs)
            else:
                raise
        first: float | None = None
        parts: list[str] = []
        usage = None
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if choices and choices[0].delta and choices[0].delta.content:
                    if first is None:
                        first = time.perf_counter()
                    parts.append(choices[0].delta.content)
        finally:
            # The stream stops at "[DONE]" without closing the HTTP body;
            # close it so the pooled connection is reusable.
            response = getattr(stream, "response", None)
            if response is not None:
                response.close()
        text = "".join(parts).strip()
        ttfb_ms = int(((first or time.perf_counter()) - t0) * 1000)
        if usage is not None:
            return _Completion(text, ttfb_ms, usage.prompt_tokens, usage.completion_tokens, False)
        est_in = sum(len(m["content"]) for m in kwargs["messages"]) // 4
        return _Completion(text, ttfb_ms, est_in, max(1, len(text) // 4), True)

    # -- one agent turn -----------------------------------------------------------

    def agent_turn(self, cfg, agent_system: str, agent_hist: list[dict], seed: int) -> AgentReply:
        """Run the agent's LLM for one turn, answering tool calls inline.

        A tool call is answered immediately and the agent is re-prompted, up
        to MAX_TOOL_CALLS_PER_TURN times. Everything the agent said along the
        way is joined into one spoken reply. `agent_hist` is appended in place.
        """
        t0 = time.perf_counter()
        spoken_parts: list[str] = []
        tool_calls: list[str] = []
        tokens_in = tokens_out = 0
        estimated = False
        latency: int | None = None
        for _ in range(MAX_TOOL_CALLS_PER_TURN + 1):
            window = agent_hist[-(2 * cfg.max_context_turns + 2 * MAX_TOOL_CALLS_PER_TURN):]
            before_ms = int((time.perf_counter() - t0) * 1000)
            try:
                a = self._complete(cfg.model, agent_system, window, cfg.temperature, seed, role="agent")
            except Exception as e:  # noqa: BLE001 - a refused or failed request is a silent turn on a live call
                self.last_agent_error = f"{type(e).__name__}: {str(e)[:200]}"
                a = _Completion(text="", ttfb_ms=int((time.perf_counter() - t0) * 1000), tokens_in=0, tokens_out=0, tokens_estimated=True)
                agent_hist.append({"role": "assistant", "content": ""})
                break
            tokens_in += a.tokens_in
            tokens_out += a.tokens_out
            estimated |= a.tokens_estimated
            agent_hist.append({"role": "assistant", "content": a.text})
            said = _spoken(a.text)
            if said:
                spoken_parts.append(said)
                if latency is None:
                    latency = before_ms + a.ttfb_ms
            m = TOOL_RE.search(a.text)
            if not m:
                break
            name = m.group(1)
            tool_calls.append(name)
            result = self.tool_overrides.get(name) or TOOL_RESULTS.get(name, f"error: unknown tool {name}")
            agent_hist.append({"role": "user", "content": f"[tool result] {name}: {result}"})
        if latency is None:
            latency = int((time.perf_counter() - t0) * 1000)
        return AgentReply(" ".join(spoken_parts) or "(silence)", latency, tool_calls,
                          tokens_in, tokens_out, estimated)

    def agent_system_prompt(self, program: VoiceProgram) -> str:
        s = program.config.system_prompt
        if program.tools:
            s += TOOL_INSTRUCTIONS.format(tools=", ".join(program.tools))
        return s

    def caller_system_prompt(self, scenario: Scenario) -> str:
        return CALLER_SYSTEM.format(
            goal=scenario.goal,
            style=CALLER_STYLES.get(scenario.caller_style, CALLER_STYLES["neutral"]),
            needs_human=NEEDS_HUMAN_NOTE if scenario.needs_human else "",
            hangup=HANGUP,
        )

    def protocol_verdicts(self, scenario: Scenario, turns: list[Turn], seed: int) -> dict:
        """Protocol v0.1 judge: one structured call from the pinned model."""
        from ..protocol import judge as pj

        model = self.judge_model or self.caller_model or "gpt-4o-mini"
        meta = scenario.metadata

        def complete(system: str, user: str) -> str:
            return self._complete(model, system, [{"role": "user", "content": user}], 0.0, seed).text

        return pj.judge(complete, scenario.goal, turns, needs_human=scenario.needs_human,
                        correction=bool(meta.get("correction")), identity=meta.get("identity"), model=model)

    def judge(self, scenario: Scenario, turns: list[Turn], seed: int) -> tuple[bool, str]:
        """Ask the judge model whether the caller's goal was met; returns (ok, verdict)."""
        judge_model = self.judge_model or self.caller_model or "gpt-4o-mini"
        transcript = "\n".join(
            f"{'CALLER' if tr.speaker == 'user' else 'AGENT'}: {tr.text}" for tr in turns
        )
        j = self._complete(
            judge_model,
            JUDGE_SYSTEM,
            [{"role": "user", "content": JUDGE_USER.format(
                goal=scenario.goal,
                question=JUDGE_Q_NEEDS_HUMAN if scenario.needs_human else JUDGE_Q_SOLVABLE,
                transcript=transcript or "(no turns)",
            )}],
            0.0,
            seed,
        )
        return j.text.strip().upper().startswith("YES"), j.text

    # -- the conversation ---------------------------------------------------------

    def _converse(self, program: VoiceProgram, scenario: Scenario, seed: int) -> Conversation:
        cfg = program.config
        caller_model = self.caller_model or cfg.model
        judge_model = self.judge_model or caller_model

        agent_system = self.agent_system_prompt(program)
        caller_system = self.caller_system_prompt(scenario)

        agent_hist: list[dict] = []
        caller_hist: list[dict] = [
            {"role": "user", "content": "(The line connects. State your issue to the agent.)"}
        ]
        turns: list[Turn] = []
        latencies: list[int] = []
        tool_calls: list[str] = []
        tokens_in = tokens_out = 0
        estimated = False
        hung_up = False
        t = 0

        for i in range(scenario.turns_expected):
            c = self._complete(caller_model, caller_system, caller_hist, 0.7, seed * 1000 + i)
            caller_text = c.text.replace(HANGUP, "").strip()
            if _hung_up(c.text) and (not caller_text or HANGUP_RE.match(caller_text)):
                hung_up = True
                break
            dur = _speech_ms(caller_text)
            turns.append(Turn("user", caller_text, t, t + dur))
            t += dur
            agent_hist.append({"role": "user", "content": caller_text})
            caller_hist.append({"role": "assistant", "content": caller_text})

            reply = self.agent_turn(cfg, agent_system, agent_hist, seed)
            tokens_in += reply.tokens_in
            tokens_out += reply.tokens_out
            estimated |= reply.tokens_estimated
            tool_calls.extend(reply.tool_calls)
            latency = reply.latency_ms
            spoken = reply.spoken
            latencies.append(latency)
            dur = _speech_ms(spoken)
            turns.append(Turn("agent", spoken, t + latency, t + latency + dur))
            t += latency + dur
            caller_hist.append({"role": "user", "content": spoken})
            if _hung_up(c.text):
                hung_up = True
                break

        if scenario.metadata.get("protocol"):
            verdicts = self.protocol_verdicts(scenario, turns, seed)
            completed, verdict = bool(verdicts.get("goal_achieved")), json.dumps(verdicts)
        else:
            verdicts = None
            completed, verdict = self.judge(scenario, turns, seed)

        unmeasured = ["turn_taking"]
        if self.price_input is None or self.price_output is None:
            cost = 0.0
            unmeasured.append("cost")
        else:
            cost = round(
                tokens_in / 1e6 * self.price_input + tokens_out / 1e6 * self.price_output, 6
            )

        return Conversation(
            scenario_id=scenario.id,
            turns=turns,
            task_completed=completed,
            tool_calls=tool_calls,
            response_latencies_ms=latencies,
            cost_usd=cost,
            metadata={
                "runtime": self.name,
                "caller_style": scenario.caller_style,
                "model": cfg.model,
                "caller_model": caller_model,
                "judge_model": judge_model,
                "judge": verdict,
                "verdicts": verdicts,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_estimated": estimated,
                "hung_up": hung_up,
                "timeline": "estimated from word count; latencies are measured",
                "unmeasured": unmeasured,
            },
        )

    def run(self, program: VoiceProgram, scenario: Scenario, seed: int) -> Conversation:
        # The sync client is deliberate: the runner is sequential, and the
        # asyncio client leaves httpcore generators unfinalized at event-loop
        # shutdown (logged as "generator didn't stop after athrow()" per call).
        if self._injected_client is not None:
            self._client = self._injected_client
            return self._converse(program, scenario, seed)
        self._client = self._new_client()
        try:
            return self._converse(program, scenario, seed)
        finally:
            self._client.close()
            self._client = None
