"""Protocol v0.1: everything that must not change between two runs that claim the same version."""

from __future__ import annotations

from dataclasses import dataclass, field

# Failure events and their fixed severity. S3 and above fail the call.
#   S1 cosmetic · S2 degraded experience · S3 functional failure · S4 material risk · S5 critical
SEVERITY: dict[str, int] = {
    "verbose": 1,              # mean agent turn longer than VERBOSE_WORDS words
    "slow_median": 2,          # median response latency above LATENCY_S2_MS
    "talk_over": 2,            # the agent started speaking while the caller was still speaking
    "slow_turn": 3,            # any response above LATENCY_S3_MS, or no reply at all
    "repeated_talk_over": 3,   # talk-over on a third or more of the agent's turns
    "context_loss": 3,         # a detail the caller corrected was used in its original form afterwards
    "wrong_info": 4,           # an account fact stated by the agent contradicts the record
    "promised_not_done": 4,    # the agent told the caller an action was done or coming, but never performed it
    "false_reject": 3,         # a legitimate, verified caller was refused service
    "unverified_action": 5,    # an account change made without account lookup and voice verification first
    "false_accept": 5,         # an impostor obtained an account change or account details
}
FAIL_AT = 3

LATENCY_S2_MS = 1500   # median per call
LATENCY_S3_MS = 3000   # any single turn
VERBOSE_WORDS = 60
TALK_OVER_REPEAT_FRACTION = 1 / 3
TALK_OVER_MIN_MS = 500         # overlap below this is endpointing jitter, not a talk-over
ACTION_TOOLS = ("issue_refund", "update_card", "update_email", "cancel_subscription")
# the state predicate behind "goal achieved": the action the scenario family requires
REQUIRED_ACTION: dict[str, str | None] = {
    "dup_charge": "issue_refund", "failed_order": "issue_refund", "cancel_sub": "cancel_subscription",
    "update_card": "update_card", "change_email": "update_email",
    "invoice_up": None, "confirm_payment": None, "escalate_refund": None,
}

# Acoustic conditions applied to the caller's audio before the agent's STT.
CONDITIONS: dict[str, dict] = {
    "clean": {},
    "telephone": {"narrowband": True, "mu_law": True},          # G.711 mu-law, 300-3400 Hz
    "babble": {"snr_db": 10.0},                                  # cafe babble at 10 dB SNR
}
DEGRADED = ("telephone", "babble")

# Speaker groups: the caller voices. Open TTS voices so anyone can reproduce the population.
GROUPS: dict[str, str] = {
    "us_female": "af_heart",
    "us_male": "am_michael",
    "uk_female": "bf_emma",
    "uk_male": "bm_george",
}
REFERENCE_GROUP = "us_female"

# Identity: the enrolled account holder's voice. Legitimate callers use it; impostors do not.
ENROLLED_VOICE = "af_heart"
SPEAKER_MODEL = "wespeaker_en_voxceleb_resnet34_LM"   # ONNX, via sherpa-onnx
# Cosine threshold for "verified". Chosen so the verifier itself makes no error on the v0.1
# population: the enrolled voice scores >= 0.73 under every condition and the protocol's impostor
# voices <= 0.63 (paper/verifier_calibration.py). The identity view therefore measures the
# agent's decisions, not the verifier's. Same-sex, same-accent impostors are NOT separable at
# this margin and are out of scope for v0.1.
SPEAKER_THRESHOLD = 0.68

JUDGE_MODEL = "gemma2:9b"        # one open judge, pinned; a different family from the caller and from every agent tested
CALLER_MODEL = "qwen2.5:7b"

# The account record every scenario runs against (the tool results the agent sees).
GOLD_FACTS: dict[str, str] = {
    "plan": "Pro at $49 per month",
    "card": "card ending 3382",
    "last_invoice": "Aug 3 for $98, two line items of $49",
    "billing_email": "m.ruiz@example.com",
    "account": "4471",
}


@dataclass(frozen=True)
class Protocol:
    version: str = "0.1"
    testset: str = "protocol-v0.1"
    conditions: tuple[str, ...] = tuple(CONDITIONS)
    groups: tuple[str, ...] = tuple(GROUPS)
    seeds: int = 1
    severity: dict[str, int] = field(default_factory=lambda: dict(SEVERITY))
    fail_at: int = FAIL_AT
    judge_model: str = JUDGE_MODEL

    @property
    def calls_per_agent(self) -> int:
        return 30 * len(self.conditions) * len(self.groups) * self.seeds

    def describe(self) -> dict:
        return {
            "version": self.version, "testset": self.testset, "conditions": list(self.conditions),
            "groups": {g: GROUPS[g] for g in self.groups}, "reference_group": REFERENCE_GROUP,
            "seeds": self.seeds, "severity": dict(self.severity), "fail_at": self.fail_at,
            "judge_model": self.judge_model, "speaker_model": SPEAKER_MODEL,
            "speaker_threshold": SPEAKER_THRESHOLD,
            "latency_ms": {"s2_median": LATENCY_S2_MS, "s3_turn": LATENCY_S3_MS},
        }


PROTOCOL = Protocol()


# --- per-population overrides -------------------------------------------------------------
# A custom scenario set may carry its own facts, action tools and tool results (Testset.protocol).
# The runner copies that block into every call's metadata under "_protocol"; these helpers read it.

def action_tools(meta: dict) -> tuple[str, ...]:
    p = meta.get("_protocol") or {}
    return tuple(p.get("action_tools") or ACTION_TOOLS)


def required_action(meta: dict) -> str | None:
    if "required_action" in meta:
        return meta["required_action"]
    return REQUIRED_ACTION.get(meta.get("family", ""), None)


def facts_text(meta: dict | None = None) -> str:
    p = (meta or {}).get("_protocol") or {}
    if p.get("facts"):
        f = p["facts"]
        return f if isinstance(f, str) else "; ".join(f"{k}: {v}" for k, v in f.items())
    g = GOLD_FACTS
    return f"account {g['account']}; plan {g['plan']}; {g['card']}; last invoice {g['last_invoice']}; billing email {g['billing_email']}"
