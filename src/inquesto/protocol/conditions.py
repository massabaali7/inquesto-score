"""Acoustic conditions on the caller's line, at 16 kHz float audio in [-1, 1].

telephone: 300-3400 Hz band limit with G.711 mu-law quantization (ITU-T G.711).
babble:    band-limited, amplitude-modulated noise mixed at a fixed SNR.
Pure numpy, so the population is reproducible without an audio toolkit.
"""

from __future__ import annotations

import numpy as np

SR = 16_000


def _lowpass_fft(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec[freqs > cutoff_hz] = 0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def narrowband(x: np.ndarray, mu_law: bool) -> np.ndarray:
    """A telephone line: everything above 3.4 kHz gone (8 kHz sampling), optional mu-law."""
    y = _lowpass_fft(x, 3400.0)
    if mu_law:
        mu = 255.0
        y = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)
        y = np.round(y * 127) / 127
        y = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)
    return y.astype(np.float32)


def babble(n: int, level: float, rng: np.random.Generator) -> np.ndarray:
    """Cafe-like noise: band-limited noise with slow amplitude modulation."""
    noise = rng.normal(0, 1, n).astype(np.float32)
    noise = _lowpass_fft(noise, 2500.0)
    t = np.arange(n) / SR
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)) * np.sin(2 * np.pi * 2.3 * t)
    noise = noise * mod
    return (noise / (np.abs(noise).max() or 1.0) * level).astype(np.float32)


def at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into x at the given SNR (speech RMS over noise RMS)."""
    speech_rms = float(np.sqrt(np.mean(x * x))) or 1e-3
    noise_rms = float(np.sqrt(np.mean(noise * noise))) or 1e-3
    return x + noise * (speech_rms / noise_rms) / (10 ** (snr_db / 20))
