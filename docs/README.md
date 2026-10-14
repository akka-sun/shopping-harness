# Documentation

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
5. [Final-200 Clean evaluation dataset](evaluation-dataset.md) defines the current
   curated benchmark and its update record.

[Reward v3](reward-v3.md) is the detailed specification shared by collection,
GRPO and evaluation.

The Harness GRPO replacement is documented in
[Harness self-evolution](plans/2026-09-14-harness-self-evolution-design.md).
The concrete repository implementation is listed in
[Harness implementation](plans/2026-09-15-harness-self-evolution-implementation.md).
It uses deterministic SFT probe trajectories to build single-trajectory
error-avoidance Memory and multi-trajectory procedural Skill, then injects
validated knowledge at runtime without another SFT stage. The standalone
`Harness/` package now contains the contracts, teacher-request dry-run pipeline,
artifact validation, runtime matcher and ephemeral guidance adapter. It does not
replace or modify the existing GRPO launchers until a separate rollout integration
and A/B validation are explicitly approved.
