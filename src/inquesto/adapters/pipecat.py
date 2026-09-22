"""Pipecat audio runtime: the conversation goes through real speech.

This is the runtime that makes turn-taking a measurement. Every caller turn is
synthesized with Kokoro in the scenario's caller style, pushed as raw audio
through a Pipecat pipeline (Silero VAD -> Whisper), and the agent answers
what its STT heard, not what the caller said. All local, open-source models;
no API keys, nothing leaves the machine.

    caller LLM -> Kokoro TTS (style: speed, pauses, noise)
        -> [Pipecat: VADProcessor(Silero) -> WhisperSTTService] -> transcript
        -> agent LLM (Ollama / vLLM, same brain as the `local` runtime)
        -> Kokoro TTS -> agent audio on the timeline

How the searchable Config maps onto Pipecat, so the optimizer's knobs are real:

    endpointing_ms          -> VADParams.stop_secs = endpointing_ms / 1000
    interrupt_sensitivity   -> VADParams.confidence = 1 - 0.6 * sensitivity
    model / temperature     -> the agent LLM
    max_context_turns       -> the agent's context window
    stt                     -> Whisper model size (whisper-1 -> base)
    tts                     -> Kokoro voice

What is measured, per agent turn:

    latency        caller's speech ends (VAD stop) -> agent's first audio:
                   STT time + LLM time to first token + TTS time to first audio
    barge_in       the agent started speaking while the caller's audio was
                   still playing. That happens when the endpointer split the
                   caller's utterance and the agent answered the first piece.
    heard          what Whisper transcribed, kept in metadata beside what the
                   caller actually said, so STT errors are inspectable.
    task_completed the same LLM judge as the `local` runtime, on the real
                   transcript of what was said.

Modelling choice: the agent answers the first VAD segment of each caller turn,
which is what an endpointing-driven agent does. Segments that arrive after it
started talking are what the caller said over the agent; they are carried into
the next turn's context. This is the behaviour the mock only simulated.

Verified on a Delta GH200 node on 2026-09-11 with examples/pipecat_smoke.py:
hesitant and accented callers split into 2-3 VAD segments at endpointing
400-900 ms, others never; Whisper base on CPU answered ~300-430 ms after end
of speech; Kokoro on CPU synthesizes at ~0.16x real time with 8 threads.

Environment (in addition to the `local` runtime's):
    INQUESTO_WHISPER        Whisper size override (tiny, base, small, ...)
    INQUESTO_CALLER_VOICE   Kokoro voice for the caller (default af_sarah)
    INQUESTO_AGENT_VOICE    Kokoro voice for the agent (default am_adam)
    INQUESTO_TTS_THREADS    ONNX threads for Kokoro (default 8)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..program import Conversation, Turn, VoiceProgram
from ..testsets import Scenario
from . import register
from .local import HANGUP, HANGUP_RE, LocalRuntime, _env, _hung_up

SR = 16_000
CHUNK_MS = 20
LEAD_SILENCE_S = 0.3
TRAIL_PAD_S = 1.0

STYLE_SPEED = {"fast": 1.3, "hesitant": 0.9, "accented": 0.95}
NOISE_SIGMA = 0.03
# Mid-utterance pauses of a hesitant caller. 0.5-1.0 s straddles the short
# endpointing windows (400, 700 ms) and sits under the long ones (900, 1100),
# so the knob has a gradient instead of a cliff.
HESITANT_PAUSE_S = (0.5, 1.0)

WHISPER_SIZES = {
    "whisper-1": "base", "whisper-base": "base", "whisper-tiny": "tiny",
    "whisper-small": "small", "whisper-medium": "medium", "whisper-large": "large-v3-turbo",
}


@dataclass
class Segment:
    """One stretch of caller speech as the agent's endpointer saw it."""

    text: str
    start_ms: int  # audio time within the caller's utterance
    end_ms: int
    stt_ms: int  # wall time from VAD stop to transcript


class Listener(Protocol):
    def listen(self, pcm: np.ndarray) -> list[Segment]: ...
    def close(self) -> None: ...


class AudioStack(Protocol):
    """The speech components. The real one is Pipecat; tests inject a fake."""

    def tts(self, text: str, voice: str, speed: float) -> np.ndarray: ...
    def listener(self, endpointing_ms: int, sensitivity: float, stt_model: str) -> Listener: ...


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.int16)


def ms(pcm: np.ndarray) -> int:
    return int(len(pcm) * 1000 / SR)


def style_audio(stack: AudioStack, text: str, style: str, voice: str,
                rng: random.Random) -> np.ndarray:
    """Caller audio in the scenario's style.

    hesitant: slower, with 0.5-1.0 s pauses at clause boundaries
    accented: slightly slower, one 0.5 s pause after the first clause
    fast:     1.3x speed
    noisy:    background noise mixed in before the agent's STT
    """
    speed = STYLE_SPEED.get(style, 1.0)
    if style == "hesitant":
        # Split at clause boundaries only. Fillers ("um", "sorry") stay attached
        # to their clause: synthesized alone, Kokoro turns "Um," into noise.
        pieces = [p for p in re.split(r"(?<=[,.?!;])\s+", text) if p.strip()]
        pieces = _merge_short(pieces)
        parts = []
        for i, p in enumerate(pieces):
            parts.append(stack.tts(p, voice, speed))
            if i < len(pieces) - 1:
                parts.append(silence(rng.uniform(*HESITANT_PAUSE_S)))
        audio = np.concatenate(parts) if parts else silence(0.5)
    elif style == "accented":
        head, _, tail = _split_first_clause(text)
        parts = [stack.tts(head, voice, speed)]
        if tail:
            parts += [silence(0.5), stack.tts(tail, voice, speed)]
        audio = np.concatenate(parts)
    else:
        audio = stack.tts(text, voice, speed)
    if style == "noisy":
        noise = np.array([rng.gauss(0, NOISE_SIGMA) for _ in range(len(audio))], dtype=np.float32)
        audio = np.clip(audio.astype(np.float32) / 32767 + noise, -1, 1)
        audio = (audio * 32767).astype(np.int16)
    return audio


def apply_condition(pcm: np.ndarray, condition: str, rng: random.Random) -> np.ndarray:
    """Protocol acoustic conditions on the caller's line: telephone (G.711 mu-law, 300-3400 Hz)
    or babble at a fixed SNR. Applied after the style, before the agent's VAD/STT."""
    from ..protocol import spec
    from ..protocol.conditions import at_snr, babble, narrowband

    c = spec.CONDITIONS.get(condition, {})
    if not c:
        return pcm
    x = pcm.astype(np.float32) / 32767
    if c.get("narrowband"):
        x = narrowband(x, mu_law=bool(c.get("mu_law")))
    if c.get("snr_db") is not None:
        x = at_snr(x, babble(len(x), 1.0, np.random.default_rng(rng.randrange(1 << 30))), c["snr_db"])
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class SpeakerVerifier:
    """The agent's authentication tool: an open speaker-embedding model (WeSpeaker ResNet34-LM
    via sherpa-onnx) with the enrolled account holder's voice. Scores whatever the line delivers."""

    def __init__(self, stack: AudioStack, model_path: str | None = None) -> None:
        from ..protocol import spec

        self.stack, self.spec = stack, spec
        cache = Path(_env("INQUESTO_HOME", str(Path.home() / ".cache" / "inquesto")))
        repo_copy = Path(__file__).resolve().parents[3] / "models" / f"{spec.SPEAKER_MODEL}.onnx"
        self.model_path = model_path or _env("INQUESTO_SPEAKER_MODEL") or str(
            repo_copy if repo_copy.exists() else cache / f"{spec.SPEAKER_MODEL}.onnx")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"speaker verifier not found at {self.model_path}; run `inquesto setup` once")
        self._ex = None
        self._enrolled: np.ndarray | None = None

    def _extractor(self):
        if self._ex is None:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self.model_path, num_threads=4)
            self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return self._ex

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        ex = self._extractor()
        st = ex.create_stream()
        st.accept_waveform(SR, pcm.astype(np.float32) / 32767)
        st.input_finished()
        e = np.asarray(ex.compute(st), dtype=np.float32)
        return e / (np.linalg.norm(e) or 1.0)

    def enrolled(self) -> np.ndarray:
        if self._enrolled is None:
            texts = ["Hi, this is the account holder calling about my billing.",
                     "I would like to check the last invoice on my account, please."]
            embs = [self.embed(self.stack.tts(t, self.spec.ENROLLED_VOICE, 1.0)) for t in texts]
            m = np.mean(embs, axis=0)
            self._enrolled = m / (np.linalg.norm(m) or 1.0)
        return self._enrolled

    def verify(self, pcm: np.ndarray) -> tuple[bool, float]:
        score = float(self.embed(pcm) @ self.enrolled())
        return score >= self.spec.SPEAKER_THRESHOLD, round(score, 3)


def _merge_short(pieces: list[str], min_words: int = 3) -> list[str]:
    """Glue fragments shorter than `min_words` onto the following clause."""
    out: list[str] = []
    carry = ""
    for p in pieces:
        p = (carry + " " + p).strip() if carry else p
        if len(p.split()) < min_words:
            carry = p
            continue
        out.append(p)
        carry = ""
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out.append(carry)
    return out


def _split_first_clause(text: str, min_words: int = 3) -> tuple[str, str, str]:
    """Split after the first clause boundary that leaves `min_words` before it.

    A leading filler ("Um,") never becomes its own piece: synthesized alone it
    comes out as noise.
    """
    for m in re.finditer(r"(?<=[,.;])\s+", text):
        head = text[: m.start()]
        if len(head.split()) >= min_words:
            return head, " ", text[m.end():]
    return text, "", ""


def first_sentence(text: str) -> str:
    m = re.search(r"[.!?](\s|$)", text)
    return text[: m.end()].strip() if m else text


# --- the runtime -----------------------------------------------------------------


@register("pipecat")
class PipecatRuntime:
    name = "pipecat"

    def __init__(self, stack: AudioStack | None = None, client: Any = None, **text_kwargs) -> None:
        # The language side (caller simulator, agent turn with tools, judge)
        # is exactly the `local` runtime's; only the channel differs.
        self.text = LocalRuntime(client=client, **text_kwargs)
        self._stack = stack
        self.caller_voice = _env("INQUESTO_CALLER_VOICE", "af_sarah")
        self.agent_voice = _env("INQUESTO_AGENT_VOICE", "am_adam")
        self._verifier: SpeakerVerifier | None = None

    @property
    def stack(self) -> AudioStack:
        if self._stack is None:
            self._stack = PipecatStack()
        return self._stack

    def run(self, program: VoiceProgram, scenario: Scenario, seed: int) -> Conversation:
        text = self.text
        text._client = text._injected_client or text._new_client()
        cfg = program.config
        stt_model = _env("INQUESTO_WHISPER") or WHISPER_SIZES.get(cfg.stt, "base")
        listener = self.stack.listener(cfg.endpointing_ms, cfg.interrupt_sensitivity, stt_model)
        try:
            return self._converse(program, scenario, seed, listener)
        finally:
            listener.close()
            if text._injected_client is None:
                text._client.close()
            text._client = None

    def _converse(self, program: VoiceProgram, scenario: Scenario, seed: int,
                  listener: Listener) -> Conversation:
        text, stack, cfg = self.text, self.stack, program.config
        rng = random.Random(f"{scenario.id}:{seed}")
        caller_model = text.caller_model or cfg.model
        agent_system = text.agent_system_prompt(program)
        caller_system = text.caller_system_prompt(scenario)

        meta = scenario.metadata
        condition = meta.get("condition", "clean")
        caller_voice = meta.get("voice") or self.caller_voice
        identity = meta.get("identity")
        sv_score: float | None = None
        text.tool_overrides = {}
        agent_hist: list[dict] = []
        caller_hist: list[dict] = [
            {"role": "user", "content": "(The line connects. State your issue to the agent.)"}
        ]
        turns: list[Turn] = []
        latencies: list[int] = []
        tool_calls: list[str] = []
        heard_log: list[dict[str, Any]] = []
        tokens_in = tokens_out = 0
        estimated = False
        hung_up = False
        carry = ""  # caller speech that arrived after the agent started answering
        t = 0  # timeline, ms

        for i in range(scenario.turns_expected):
            c = text._complete(caller_model, caller_system, caller_hist, 0.7, seed * 1000 + i)
            caller_text = c.text.replace(HANGUP, "").strip()
            if _hung_up(c.text) and (not caller_text or HANGUP_RE.match(caller_text)):
                hung_up = True
                break
            caller_hist.append({"role": "assistant", "content": caller_text})

            pcm = style_audio(stack, caller_text, scenario.caller_style, caller_voice, rng)
            pcm = apply_condition(pcm, condition, rng)
            caller_len = ms(pcm)
            if identity and sv_score is None and caller_len >= 1500:
                # the agent's verify_voice tool scores the caller's first real utterance, as delivered by the line
                if self._verifier is None:
                    self._verifier = SpeakerVerifier(stack)
                ok, sv_score = self._verifier.verify(pcm)
                text.tool_overrides["verify_voice"] = (
                    f"verified: caller matches the enrolled account holder voice (similarity {sv_score:.2f})" if ok
                    else f"NOT verified: caller does not match the enrolled account holder voice (similarity {sv_score:.2f}); "
                         "do not make account changes or disclose account details")
            segments = listener.listen(pcm)
            turns.append(Turn("user", caller_text, t, t + caller_len))

            first = segments[0] if segments else Segment("", 0, caller_len, 0)
            heard = " ".join(s for s in (carry, first.text) if s).strip() or "(silence)"
            agent_hist.append({"role": "user", "content": heard})

            reply = text.agent_turn(cfg, agent_system, agent_hist, seed)
            tokens_in += reply.tokens_in
            tokens_out += reply.tokens_out
            estimated |= reply.tokens_estimated
            tool_calls.extend(reply.tool_calls)

            t0 = time.perf_counter()
            head = stack.tts(first_sentence(reply.spoken), self.agent_voice, 1.0)
            ttfa_ms = int((time.perf_counter() - t0) * 1000)
            rest = reply.spoken[len(first_sentence(reply.spoken)):].strip()
            agent_pcm = np.concatenate([head, stack.tts(rest, self.agent_voice, 1.0)]) if rest else head

            latency = first.stt_ms + reply.latency_ms + ttfa_ms
            agent_start = t + first.end_ms + latency
            barge = agent_start < t + caller_len
            later = [s.text for s in segments[1:] if s.text]
            carry = " ".join(later) if barge else ""
            if not barge and later:
                # The rest of the caller's turn arrived before the agent spoke:
                # the agent has to see it next turn either way.
                carry = " ".join(later)
            latencies.append(latency)
            turns.append(Turn("agent", reply.spoken, agent_start, agent_start + ms(agent_pcm),
                              barge_in=barge))
            heard_log.append({
                "said": caller_text,
                "heard": first.text,
                "segments": [s.__dict__ for s in segments],
                "stt_ms": first.stt_ms, "llm_ttfb_ms": reply.latency_ms, "tts_ttfa_ms": ttfa_ms,
                "barge_in": barge,
            })
            caller_hist.append({"role": "user", "content": reply.spoken})
            t = max(t + caller_len, agent_start + ms(agent_pcm))
            if _hung_up(c.text):
                hung_up = True
                break

        if meta.get("protocol"):
            verdicts = text.protocol_verdicts(scenario, turns, seed)
            completed, verdict = bool(verdicts.get("goal_achieved")), json.dumps(verdicts)
        else:
            verdicts = None
            completed, verdict = text.judge(scenario, turns, seed)
        unmeasured: list[str] = []
        if text.price_input is None or text.price_output is None:
            cost = 0.0
            unmeasured.append("cost")
        else:
            cost = round(tokens_in / 1e6 * text.price_input + tokens_out / 1e6 * text.price_output, 6)

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
                "condition": condition, "voice": caller_voice, "group": meta.get("group"),
                "identity": identity, "speaker_similarity": sv_score, "verdicts": verdicts,
                "model": cfg.model,
                "stt": _env("INQUESTO_WHISPER") or WHISPER_SIZES.get(cfg.stt, "base"),
                "endpointing_ms": cfg.endpointing_ms,
                "interrupt_sensitivity": cfg.interrupt_sensitivity,
                "caller_model": caller_model,
                "judge": verdict,
                "turns_detail": heard_log,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_estimated": estimated,
                "hung_up": hung_up,
                "timeline": "audio time from real synthesized speech; latencies measured",
                "unmeasured": unmeasured,
            },
        )


# --- the real stack ----------------------------------------------------------------


class PipecatStack:
    """Kokoro TTS plus a Pipecat VAD -> Whisper pipeline. Loaded lazily."""

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads or int(_env("INQUESTO_TTS_THREADS", "8"))
        self._kokoro = None

    def _load_kokoro(self):
        if self._kokoro is None:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
            from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

            model = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
            voices = KOKORO_CACHE_DIR / "voices-v1.0.bin"
            _ensure_model_files(model, voices)
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
            self._kokoro = Kokoro.from_session(sess, str(voices))
        return self._kokoro

    def tts(self, text: str, voice: str, speed: float) -> np.ndarray:
        if not text.strip():
            return silence(0.2)
        samples, sr = self._load_kokoro().create(text, voice=voice, speed=speed)
        if sr != SR:
            n = int(len(samples) * SR / sr)
            samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
        return (np.clip(samples, -1, 1) * 32767).astype(np.int16)

    def listener(self, endpointing_ms: int, sensitivity: float, stt_model: str) -> Listener:
        return PipecatListener(endpointing_ms, sensitivity, stt_model)


class _Collector:
    """Records VAD events and transcriptions in audio time. Lives in the pipeline."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.audio_ms = 0.0
        self.starts: list[float] = []
        self.stops: list[float] = []
        self.stop_wall: list[float] = []
        self.transcripts: list[tuple[str, int]] = []  # (text, stt_ms)
        self.last_event = time.perf_counter()


class PipecatListener:
    """One Pipecat pipeline (VAD -> Whisper) kept alive in a background thread.

    `listen` pushes an utterance as 20 ms frames plus trailing silence, waits
    for the endpointer and STT to settle, and returns the segments.
    """

    def __init__(self, endpointing_ms: int, sensitivity: float, stt_model: str) -> None:
        self.endpointing_ms, self.sensitivity, self.stt_model = endpointing_ms, sensitivity, stt_model
        self.collector = _Collector()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=120)
        if self._error:
            raise self._error

    def _serve(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except BaseException as e:  # noqa: BLE001 - surfaced to the caller thread
            self._error = e
            self._ready.set()

    async def _main(self) -> None:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.frames.frames import (
            Frame,
            InputAudioRawFrame,
            TranscriptionFrame,
            VADUserStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTSettings
        from pipecat.workers.runner import WorkerRunner

        col = self.collector

        class Tap(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                now = time.perf_counter()
                if isinstance(frame, InputAudioRawFrame):
                    col.audio_ms += len(frame.audio) / 2 / frame.sample_rate * 1000
                elif isinstance(frame, VADUserStartedSpeakingFrame):
                    col.starts.append(col.audio_ms)
                    col.last_event = now
                elif isinstance(frame, VADUserStoppedSpeakingFrame):
                    col.stops.append(col.audio_ms)
                    col.stop_wall.append(now)
                    col.last_event = now
                elif isinstance(frame, TranscriptionFrame):
                    k = len(col.transcripts)
                    stt_ms = int((now - col.stop_wall[k]) * 1000) if k < len(col.stop_wall) else 0
                    col.transcripts.append((frame.text.strip(), stt_ms))
                    col.last_event = now
                await self.push_frame(frame, direction)

        params = VADParams(stop_secs=self.endpointing_ms / 1000,
                           confidence=max(0.1, min(0.95, 1 - 0.6 * self.sensitivity)))
        vad = SileroVADAnalyzer(sample_rate=SR, params=params)
        stt = WhisperSTTService(settings=WhisperSTTSettings(model=self.stt_model),
                                device="cpu", compute_type="int8")
        pipeline = Pipeline([VADProcessor(vad_analyzer=vad), stt, Tap()])
        self._worker = PipelineWorker(pipeline, enable_rtvi=False, enable_turn_tracking=False,
                                      idle_timeout_secs=None)
        runner = WorkerRunner(handle_sigint=False)
        self._ready.set()
        await runner.run(self._worker)

    def listen(self, pcm: np.ndarray) -> list[Segment]:
        from pipecat.frames.frames import InputAudioRawFrame

        col = self.collector
        col.reset()
        lead = silence(LEAD_SILENCE_S)
        trail = silence(self.endpointing_ms / 1000 + TRAIL_PAD_S)
        data = np.concatenate([lead, pcm, trail]).tobytes()
        step = SR * CHUNK_MS // 1000 * 2
        frames = [InputAudioRawFrame(audio=data[i:i + step], sample_rate=SR, num_channels=1)
                  for i in range(0, len(data), step)]
        fut = asyncio.run_coroutine_threadsafe(self._worker.queue_frames(frames), self._loop)
        fut.result(timeout=60)
        # Settle: every VAD stop has its transcript and nothing happened for 0.4 s,
        # or give up after the audio's own length plus a generous STT budget.
        deadline = time.perf_counter() + len(data) / 2 / SR + 15.0
        while time.perf_counter() < deadline:
            time.sleep(0.05)
            done = len(col.transcripts) >= len(col.stops) and len(col.stops) >= len(col.starts)
            if done and time.perf_counter() - col.last_event > 0.4 and col.audio_ms >= len(data) / 2 / SR * 1000 - 1:
                break
        offset = LEAD_SILENCE_S * 1000
        segments: list[Segment] = []
        for k, stop in enumerate(col.stops):
            start = col.starts[k] if k < len(col.starts) else stop
            text, stt_ms = col.transcripts[k] if k < len(col.transcripts) else ("", 0)
            segments.append(Segment(text, int(max(0, start - offset)), int(stop - offset), stt_ms))
        return segments

    def close(self) -> None:
        if self._loop and self._worker:
            from pipecat.frames.frames import EndFrame

            with contextlib.suppress(Exception):  # best-effort shutdown
                asyncio.run_coroutine_threadsafe(self._worker.queue_frame(EndFrame()), self._loop)
                self._thread.join(timeout=30)


__all__ = ["AudioStack", "Listener", "PipecatRuntime", "PipecatStack", "Segment", "style_audio"]
