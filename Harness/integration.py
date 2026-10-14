"""Small adapter for injecting ephemeral Harness guidance into rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from .runtime import RuntimeGuidance


def with_guidance(
    messages: Sequence[Mapping],
    guidance: RuntimeGuidance,
) -> list[dict]:
    """Return a request-only message copy with the current guidance appended.

    The returned message is intentionally ephemeral: callers should pass this
    list to ``client.complete`` but keep their canonical trajectory history
    unchanged.  This prevents old Skill/Memory text from accumulating over turns
    and leaves system prompts, tool schemas, guards and Reward untouched.
    """
    result = [deepcopy(dict(message)) for message in messages]
    if guidance.message:
        result.append({"role": "user", "content": guidance.message})
    return result


def complete_with_guidance(client, messages, tools, bank, state, *, memory_limit: int = 2):
    """Match current state and make one completion using ephemeral guidance.

    This is deliberately an adapter rather than a modification of the existing
    GRPO/evaluation rollout.  It is suitable for a future Harness-enabled rollout
    and easy to unit-test with a fake client.
    """
    guidance = bank.match(state, memory_limit=memory_limit)
    response = client.complete(with_guidance(messages, guidance), tools)
    return response, guidance

