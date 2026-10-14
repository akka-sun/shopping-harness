"""Teacher-induced Memory/Skill harness for SFT probe trajectories.

The package is deliberately independent from the GRPO training stack.  It only
consumes normalized, actor-visible trajectories and produces validated knowledge
artifacts plus runtime guidance.
"""

from .contracts import (
    MEMORY_SCHEMA_VERSION,
    SKILL_SCHEMA_VERSION,
    HarnessValidationError,
    validate_memory,
    validate_skill,
)
from .runtime import HarnessState, KnowledgeBank, RuntimeGuidance, build_guidance
from .prompts import actor_visible_trajectory
from .integration import complete_with_guidance, with_guidance

__all__ = [
    "HarnessState",
    "KnowledgeBank",
    "RuntimeGuidance",
    "HarnessValidationError",
    "MEMORY_SCHEMA_VERSION",
    "SKILL_SCHEMA_VERSION",
    "validate_memory",
    "validate_skill",
    "build_guidance",
    "actor_visible_trajectory",
    "with_guidance",
    "complete_with_guidance",
]
