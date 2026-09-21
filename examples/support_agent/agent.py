"""The reference agent used by the Inquesto demo and benchmark."""

from inquesto import VoiceProgram


class SupportAgent(VoiceProgram):
    task = (
        "You are a billing support agent. Resolve the caller's billing question "
        "in under three minutes, confirming account details before any change."
    )
    constraints = [
        "never invent refund policy",
        "confirm the account before making changes",
        "let the caller finish speaking",
    ]
    tools = ["lookup_account", "issue_refund"]
