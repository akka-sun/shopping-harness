# Evidence Curriculum GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic Fresh SFT difficulty probing, public-only trajectory credit assignment, and a checkpointable 30/70 difficulty/failure-aware GRPO task sampler without changing the baseline entrypoint or training budget.

**Architecture:** All scoring and sampling logic lives in focused `shopping_grpo.training.grpo` modules. The existing AgentLoop emits public evidence and turn spans, while a version-pinned veRL patch invokes task selection, group-stat updates, post-GRPO token weighting, and checkpoint serialization. Existing Reward v3 group filtering remains independent and authoritative.

**Tech Stack:** Python 3.10+, pytest, PyTorch tensors at the veRL boundary, Hydra/OmegaConf configuration, veRL 0.8.0, ShopSimulator Environment v2.1, Reward v3.

**Spec:** `docs/superpowers/specs/2026-08-23-evidence-curriculum-design.md`

## Global Constraints

- Preserve the only supported workflow: `Baseline → SFT → GRPO → Evaluation`.
- Runtime contracts remain ShopSimulator Environment v2.1, Reward v3, observation v2, and tool schema v2.
- Strict success remains a complete `gold_purchase` with `reward_valid=true`.
- Training and validation task IDs must have zero overlap with `data/evaluation/tasks.jsonl` and with one another.
- Do not change Reward v3, raw GRPO advantage, actor loss, PPO clipping, KL, rollout log probabilities, or response-mask semantics.
- Keep `rollout.n`, total optimizer steps, `max_num_gen_batches=3`, and `max_environment_steps=35` equal to the baseline.
- Do not run GRPO training, merge models, or run the 200-task Evaluation while implementing this plan.
- Preserve unrelated dirty-worktree changes; stage and commit only files belonging to the current task.

---

## File Map

**New core files**

- `src/shopping_grpo/training/grpo/difficulty/instruction.py`: deterministic public-instruction features and score.
- `src/shopping_grpo/training/grpo/difficulty/manifest.py`: Probe record validation, quantile boundaries, overlap checks, hashes, and atomic artifacts.
- `src/shopping_grpo/training/grpo/difficulty/probe.py`: deterministic one-rollout-per-task Probe coordinator.
- `src/shopping_grpo/training/grpo/evidence.py`: public state snapshots and evidence events.
- `src/shopping_grpo/training/grpo/trajectory_credit.py`: turn scores, bounded conservation, and token projection.
- `src/shopping_grpo/training/grpo/failure_types.py`: Reward/runtime normalization.
- `src/shopping_grpo/training/grpo/curriculum/state.py`: online statistics and versioned checkpoint state.
- `src/shopping_grpo/training/grpo/curriculum/priority.py`: fixed v1 learnability formula.
- `src/shopping_grpo/training/grpo/curriculum/sampler.py`: phase selection and deterministic mixed sampling.
- `scripts/probe_grpo_difficulty.py`: Probe audit and execution entrypoint.
- `scripts/train_grpo_evidence_curriculum.py`: isolated feature-enabling GRPO entrypoint.
- `configs/grpo_evidence_curriculum.yaml`: explicit feature configuration.

**Modified integration files**

- `src/shopping_grpo/training/grpo/adapter/runtime.py`: trajectory-local evidence/turn metadata state.
- `src/shopping_grpo/training/grpo/adapter/tools.py`: before/after public snapshots and evidence recording.
- `src/shopping_grpo/training/grpo/adapter/agent_loop.py`: stable Assistant turn spans and safe exported metadata.
- `configs/grpo.yaml`: add both new namespaces with `enable: false` only.
- `scripts/check_grpo_runtime.py`: validate feature configs, manifest, and patch marker.
- `patches/verl-0.8.0-shopping-dynamic-sampling.patch`: thin sampler/credit/checkpoint hooks.
- `scripts/apply_verl_dynamic_sampling_patch.py`: update marker and verified patched hash after patch regeneration.
- `docs/grpo.md`: Probe and evidence-curriculum usage, dry-run, artifacts, resume rules, and diagnostics.

---

### Task 1: Deterministic Public-Instruction Difficulty Scorer

**Files:**

- Create: `src/shopping_grpo/training/grpo/difficulty/__init__.py`
- Create: `src/shopping_grpo/training/grpo/difficulty/instruction.py`
- Create: `tests/test_grpo_instruction_difficulty.py`

**Interfaces:**

- Consumes: public instruction `str` only.
- Produces: `InstructionDifficulty(score: float, features: Mapping[str, int | float | bool], scorer_version: str)` and `score_public_instruction(instruction: str) -> InstructionDifficulty`.

- [ ] **Step 1: Write failing tests for deterministic, bounded, public-only scoring**

```python
from shopping_grpo.training.grpo.difficulty.instruction import (
    INSTRUCTION_SCORER_VERSION,
    score_public_instruction,
)


def test_instruction_score_is_deterministic_and_bounded():
    text = "Buy a Sony WH-1000XM5 under $350 in black, compatible with Android."
    first = score_public_instruction(text)
    second = score_public_instruction(text)
    assert first == second
    assert 0.0 <= first.score <= 1.0
    assert first.scorer_version == INSTRUCTION_SCORER_VERSION
    assert first.features["has_budget"] is True
    assert first.features["brand_model_constraints"] >= 1
    assert first.features["compatibility_constraints"] == 1
    assert text not in repr(first.features)


def test_instruction_length_is_only_a_weak_capped_feature():
    simple = score_public_instruction("Buy a red mug.")
    padded = score_public_instruction("Buy a red mug. " + "Please consider carefully. " * 100)
    constrained = score_public_instruction(
        "Buy a red Contigo mug under $25, 16 oz, dishwasher safe, compatible with car holders."
    )
    assert padded.score - simple.score <= 0.05
    assert constrained.score > padded.score
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `pytest -q tests/test_grpo_instruction_difficulty.py`  
Expected: FAIL with `ModuleNotFoundError` for `shopping_grpo.training.grpo.difficulty`.

- [ ] **Step 3: Implement the versioned scorer and explicit feature weights**

Implement immutable dataclasses and these v1 feature caps/weights in `instruction.py`:

```python
INSTRUCTION_SCORER_VERSION = "grpo-public-instruction-v1"
FEATURE_WEIGHTS = {
    "constraint_categories": 0.25,
    "numeric_constraints": 0.15,
    "brand_model_constraints": 0.15,
    "option_axes": 0.15,
    "compatibility_constraints": 0.15,
    "joint_requirement": 0.10,
    "length_weak": 0.05,
}
FEATURE_CAPS = {
    "constraint_categories": 6,
    "numeric_constraints": 3,
    "brand_model_constraints": 2,
    "option_axes": 4,
    "compatibility_constraints": 2,
}

@dataclass(frozen=True)
class InstructionDifficulty:
    score: float
    features: Mapping[str, int | float | bool]
    scorer_version: str = INSTRUCTION_SCORER_VERSION

def score_public_instruction(instruction: str) -> InstructionDifficulty:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    features = extract_public_features(instruction)
    normalized = {
        name: min(float(features[name]) / FEATURE_CAPS[name], 1.0)
        for name in FEATURE_CAPS
    }
    normalized["joint_requirement"] = float(bool(features["joint_requirement"]))
    normalized["length_weak"] = min(len(instruction.split()) / 200.0, 1.0)
    score = sum(FEATURE_WEIGHTS[name] * normalized[name] for name in FEATURE_WEIGHTS)
    return InstructionDifficulty(
        score=min(max(score, 0.0), 1.0),
        features={**features, **{f"normalized_{k}": v for k, v in normalized.items()}},
    )
```

Use compiled, version-controlled patterns for currency/numeric comparators, brand/model tokens, color/size/capacity/material options, compatibility phrases, and conjunctions. Reject empty/non-string instructions. Normalize each count by its cap, compute the exact weighted sum, and clamp only for floating-point safety.

- [ ] **Step 4: Run focused tests and static checks**

Run: `pytest -q tests/test_grpo_instruction_difficulty.py && .venv/bin/ruff check src/shopping_grpo/training/grpo/difficulty tests/test_grpo_instruction_difficulty.py`  
Expected: PASS.

- [ ] **Step 5: Commit the scorer**

```bash
git add src/shopping_grpo/training/grpo/difficulty tests/test_grpo_instruction_difficulty.py
git commit -m "feat: add public GRPO difficulty scorer"
```

---

### Task 2: Probe Scores, Split Isolation, Quantiles, and Artifacts

**Files:**

- Create: `src/shopping_grpo/training/grpo/difficulty/manifest.py`
- Create: `tests/test_grpo_difficulty_manifest.py`

**Interfaces:**

- Consumes: task rows, runtime counters, Probe validity, train/validation/evaluation task IDs.
- Produces: `compute_probe_scores(...)`, `fit_train_boundaries(...)`, `assign_category(...)`, `validate_task_isolation(...)`, `write_probe_artifacts(...)`.

- [ ] **Step 1: Write failing formula and boundary tests**

```python
import pytest
from shopping_grpo.training.grpo.difficulty.manifest import (
    ProbeCounters,
    assign_categories,
    compute_probe_scores,
    fit_train_boundaries,
    validate_task_isolation,
)


def test_probe_formula_is_exact_and_bounded():
    scores = compute_probe_scores(
        task_difficulty=0.8,
        counters=ProbeCounters(
            action_steps=35,
            action_attempt_count=10,
            repeat_action_count=2,
            guard_count=3,
            max_environment_steps=35,
        ),
    )
    assert scores.step_score == 1.0
    assert scores.repeat_ratio == 0.2
    assert scores.guard_score == 1.0
    assert scores.probe_difficulty == pytest.approx(0.79, abs=1e-8)


def test_train_boundaries_do_not_split_ties_or_read_validation():
    train = [(1, 0.1), (2, 0.2), (3, 0.2), (4, 0.8), (5, 0.9), (6, 1.0)]
    boundaries = fit_train_boundaries(train)
    before = assign_categories(train, boundaries)
    after = assign_categories(train, boundaries)
    assert before == after
    assert before[2] == before[3]


def test_task_isolation_fails_closed():
    with pytest.raises(ValueError, match="overlap"):
        validate_task_isolation(train_ids={1, 2}, validation_ids={3}, evaluation_ids={2, 4})
```

- [ ] **Step 2: Run tests and confirm missing APIs**

Run: `pytest -q tests/test_grpo_difficulty_manifest.py`  
Expected: FAIL on imports.

- [ ] **Step 3: Implement immutable schemas and fixed scoring**

Define `ProbeCounters`, `ProbeScores`, `DifficultyBoundary`, `ValidProbeRecord`, and `InvalidProbeRecord`. Implement exactly:

```python
step_score = min(action_steps / max_environment_steps, 1.0)
repeat_ratio = repeat_action_count / max(action_attempt_count, 1)
guard_score = min(guard_count / 3.0, 1.0)
probe_difficulty = (
    0.45 * task_difficulty
    + 0.30 * step_score
    + 0.15 * repeat_ratio
    + 0.10 * guard_score
)
```

Reject negative counters, repeats greater than attempts, non-35 default step limits, and non-finite/out-of-range inputs.

- [ ] **Step 4: Implement deterministic train-only tied quantiles and isolation**

Sort `(score, task_id)`, locate nominal 33% and 66% positions, then move each cut between adjacent distinct scores. Raise if three non-empty categories cannot be formed without splitting ties. Apply frozen midpoint boundaries to validation. Verify unique IDs, exact expected coverage, train/validation disjointness, and Evaluation disjointness before and after record generation.

- [ ] **Step 5: Add artifact tests and atomic writers**

```python
def test_artifacts_are_separate_hashed_and_do_not_copy_instruction(tmp_path):
    paths = write_probe_artifacts(
        output_dir=tmp_path,
        valid_records=[valid_record(task_id=1, split="train")],
        invalid_records=[invalid_record(task_id=2, split="validation")],
        metadata=metadata_fixture(),
    )
    assert paths.manifest.name == "difficulty_manifest.jsonl"
    assert paths.metadata.name == "probe_metadata.json"
    assert paths.invalid.name == "invalid_probes.jsonl"
    assert "full public instruction" not in paths.manifest.read_text()
    assert load_metadata(paths.metadata)["manifest_sha256"] == sha256_file(paths.manifest)
```

Write temporary sibling files, fsync/close, then `Path.replace` each final artifact. Metadata includes all hashes, versions, decoding values, weights, boundaries, and overlap result.

- [ ] **Step 6: Run focused tests**

Run: `pytest -q tests/test_grpo_difficulty_manifest.py`  
Expected: PASS.

- [ ] **Step 7: Commit manifest logic**

```bash
git add src/shopping_grpo/training/grpo/difficulty/manifest.py tests/test_grpo_difficulty_manifest.py
git commit -m "feat: build versioned GRPO difficulty manifests"
```

---

### Task 3: Deterministic Fresh SFT Probe Entry

**Files:**

- Create: `src/shopping_grpo/training/grpo/difficulty/probe.py`
- Create: `scripts/probe_grpo_difficulty.py`
- Create: `tests/test_probe_grpo_difficulty.py`
- Modify: `tests/test_public_entrypoints.py`

**Interfaces:**

- Consumes: model directory, train/validation JSONL, Evaluation JSONL, environment URL, output directory, seed, max steps.
- Produces: `ProbeRequest`, `ProbeOutcome`, `run_probe(request, rollout_one)`, and CLI dry-run/execution audit.

- [ ] **Step 1: Write failing dry-run and coordinator tests**

```python
def test_dry_run_has_no_model_environment_or_files(monkeypatch, tmp_path):
    monkeypatch.setattr(probe_cli, "load_probe_policy", forbidden)
    monkeypatch.setattr(probe_cli, "run_probe", forbidden)
    result = probe_cli.main_for_test(valid_args(tmp_path, dry_run=True))
    assert result["decode"] == {"temperature": 0.0, "top_p": 1.0, "do_sample": False}
    assert not (tmp_path / "probe-output").exists()


def test_probe_calls_each_task_once_and_separates_invalid(tmp_path):
    calls = []
    def rollout_one(task_id, decode, seed, max_steps):
        calls.append(task_id)
        return valid_outcome(task_id) if task_id != 3 else invalid_outcome(task_id, "reward_unverifiable")
    result = run_probe(request_fixture(tmp_path), rollout_one)
    assert calls == [1, 2, 3]
    assert result.valid_task_ids == (1, 2)
    assert result.invalid_task_ids == (3,)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_probe_grpo_difficulty.py tests/test_public_entrypoints.py`  
Expected: FAIL because the entrypoint and coordinator do not exist.

- [ ] **Step 3: Implement the Probe coordinator**

In `probe.py`, define:

```python
DETERMINISTIC_DECODE = {"temperature": 0.0, "top_p": 1.0, "do_sample": False}

@dataclass(frozen=True)
class ProbeRequest:
    model: Path
    train_tasks: Path
    validation_tasks: Path
    evaluation_tasks: Path
    output_dir: Path
    raw_run_dir: Path | None
    environment_url: str
    seed: int
    max_environment_steps: int = 35

```

Implement `run_probe(request: ProbeRequest, rollout_one: Callable[[int, Mapping[str, object], int, int], ProbeOutcome]) -> ProbeRunResult`. Prevalidate isolation before the first rollout. Iterate train then validation in stable input order, exactly once. Accept `repeat_loop` and `max_steps` when terminal and Reward-valid. Mark environment errors, unverifiable Reward, incomplete terminal, schema mismatch, or context hard limit invalid. Write artifacts only after all tasks are accounted for and postvalidate isolation.

- [ ] **Step 4: Implement the CLI with a strict dry-run split**

Make `parse_args(argv=None)`, `build_audit(args)`, and `main_for_test(argv)` importable. Resolve and validate all paths before branching. Under `--dry-run`, print JSON and return before output creation, policy import, Ray initialization, HTTP calls, or AgentLoop construction. In execution mode, use the same model path and `ShoppingToolAgentLoop`/`ShopSimulatorSession` stack as GRPO; do not create a second HTTP rollout implementation.

- [ ] **Step 5: Add public-entrypoint coverage**

Extend `tests/test_public_entrypoints.py` to assert the script exists, is executable through `python scripts/probe_grpo_difficulty.py --help`, and mentions `--dry-run`, `--train-data`, `--val-data`, `--evaluation-data`, `--model`, and `--output`.

- [ ] **Step 6: Run CLI tests without executing rollouts**

Run: `pytest -q tests/test_probe_grpo_difficulty.py tests/test_public_entrypoints.py`  
Expected: PASS; no model is loaded and no environment is contacted.

- [ ] **Step 7: Commit the Probe entry**

```bash
git add src/shopping_grpo/training/grpo/difficulty/probe.py scripts/probe_grpo_difficulty.py tests/test_probe_grpo_difficulty.py tests/test_public_entrypoints.py
git commit -m "feat: add deterministic GRPO difficulty probe"
```

---

### Task 4: Public Evidence Events

**Files:**

- Create: `src/shopping_grpo/training/grpo/evidence.py`
- Create: `tests/test_grpo_evidence.py`

**Interfaces:**

- Consumes: public pre/post runtime state and action outcome.
- Produces: `PublicStateSnapshot`, `EvidenceEvent`, `capture_public_snapshot(...)`, `build_evidence_event(...)`, `validate_evidence_event(...)`.

- [ ] **Step 1: Write failing positive, negative, and privacy tests**

```python
def test_new_search_results_are_positive_and_projection_independent():
    before = snapshot(search_result_ids={"p1"})
    after = snapshot(search_result_ids={"p1", "p2"})
    event = build_evidence_event(4, before, after, action_attempted=True)
    assert event.assistant_turn_id == 4
    assert event.new_search_result_set is True
    assert event.new_product_count == 1
    assert event.no_progress is False


def test_repeat_guard_without_state_change_is_negative():
    state = snapshot()
    event = build_evidence_event(
        2, state, state, action_attempted=True,
        exact_repeat=True, guard_rejection=True, terminal=False,
    )
    assert event.exact_repeat and event.guard_rejection and event.no_progress


def test_serialized_event_has_no_hidden_or_raw_fields():
    payload = evidence_fixture().to_dict()
    forbidden = {"target_asin", "gold", "hidden_constraints", "observation", "parameters"}
    assert forbidden.isdisjoint(payload)
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `pytest -q tests/test_grpo_evidence.py`  
Expected: FAIL on import.

- [ ] **Step 3: Implement immutable public snapshots and v1 events**

Use hashed public identifiers inside the transient snapshot; serialize only counts/booleans in the event. Define `EVIDENCE_VERSION = "shopping-evidence-v1"`. `no_progress` is true only when there is no positive evidence, the public state fingerprint is unchanged, and the action is not terminal. Unknown event keys or versions fail validation.

- [ ] **Step 4: Test projection invariance explicitly**

Add a test that builds events from identical raw public states while passing two different projected renderings; the events must compare equal because projection text is not an input.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest -q tests/test_grpo_evidence.py tests/test_observation_projection.py`  
Expected: PASS.

```bash
git add src/shopping_grpo/training/grpo/evidence.py tests/test_grpo_evidence.py
git commit -m "feat: emit public trajectory evidence"
```

---

### Task 5: Bounded Conservative Trajectory Credit and Token Projection

**Files:**

- Create: `src/shopping_grpo/training/grpo/trajectory_credit.py`
- Create: `tests/test_trajectory_credit.py`

**Interfaces:**

- Consumes: scalar trajectory advantage, `EvidenceEvent` list, Assistant turn token spans, response mask.
- Produces: `compute_turn_multipliers(...) -> CreditResult` and `project_token_advantages(...) -> ProjectionResult`.

- [ ] **Step 1: Write failing direction, conservation, mask, and fallback tests**

```python
import torch

def test_positive_and_negative_advantage_reverse_responsibility_direction():
    events = [positive_event(0), repeat_event(1)]
    spans = [TurnSpan(0, 0, 2), TurnSpan(1, 2, 4)]
    positive = compute_turn_multipliers(1.0, events, spans, CreditConfig())
    negative = compute_turn_multipliers(-1.0, events, spans, CreditConfig())
    assert positive.multipliers[0] > positive.multipliers[1]
    assert negative.multipliers[0] < negative.multipliers[1]


def test_projection_preserves_mass_and_non_assistant_zero():
    result = project_token_advantages(
        trajectory_advantage=2.0,
        response_mask=torch.tensor([1, 1, 0, 1, 1, 0]),
        spans=[TurnSpan(0, 0, 2), TurnSpan(1, 3, 5)],
        events=[positive_event(0), repeat_event(1)],
        config=CreditConfig(),
    )
    assert result.token_advantages[2].item() == 0
    assert result.token_advantages[5].item() == 0
    assert result.token_advantages.sum().item() == pytest.approx(8.0, abs=1e-6)
    assert min(result.multipliers) >= 0.5
    assert max(result.multipliers) <= 1.5


@pytest.mark.parametrize("reason", ["unknown_version", "missing_event", "bad_span", "compacted"])
def test_alignment_errors_fallback_for_whole_trajectory(reason):
    result = invalid_credit_fixture(reason)
    assert result.fallback_reason == reason
    assert set(result.multipliers) == {1.0}
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_trajectory_credit.py`  
Expected: FAIL on import.

- [ ] **Step 3: Implement event scoring and bounded centering in float64**

Define:

```python
@dataclass(frozen=True)
class TurnSpan:
    assistant_turn_id: int
    start: int
    end: int

@dataclass(frozen=True)
class CreditConfig:
    positive_evidence_score: float = 1.0
    no_progress_score: float = -0.5
    exact_repeat_score: float = -1.0
    guard_rejection_score: float = -1.0
    strength: float = 0.5
    multiplier_min: float = 0.5
    multiplier_max: float = 1.5
    conservation_tolerance: float = 1e-6
```

Sum event components and clip each event score to `[-1,1]`. Reverse it for negative advantage. Center using Assistant token counts, compute the maximum globally feasible scale, then apply one residual correction without crossing bounds. Validate finite values, exact turn IDs, non-overlapping ordered spans, positive span lengths, and response-mask membership. Zero advantage returns zero token advantages.

- [ ] **Step 4: Implement whole-trajectory fallback and rolling-rate helper**

Fallback returns ordinary GRPO token advantages and a single structured reason. Add `CreditFallbackWindow(size: int, threshold: float)` whose `record(fallback: bool)` raises `RuntimeError` only after the window is full and its rate exceeds the threshold.

- [ ] **Step 5: Run tensor and pure-math tests**

Run: `pytest -q tests/test_trajectory_credit.py`  
Expected: PASS, including mass error `<1e-6`.

- [ ] **Step 6: Commit trajectory credit**

```bash
git add src/shopping_grpo/training/grpo/trajectory_credit.py tests/test_trajectory_credit.py
git commit -m "feat: add conservative trajectory credit"
```

---

### Task 6: Failure Normalization and Online State

**Files:**

- Create: `src/shopping_grpo/training/grpo/failure_types.py`
- Create: `src/shopping_grpo/training/grpo/curriculum/__init__.py`
- Create: `src/shopping_grpo/training/grpo/curriculum/state.py`
- Create: `tests/test_curriculum_state.py`

**Interfaces:**

- Consumes: validated Reward/runtime result and complete uid-group utilities.
- Produces: `normalize_failure_type(info) -> FailureType`, `CurriculumState.update_group(...)`, `to_checkpoint()`, and `from_checkpoint(...)`.

- [ ] **Step 1: Write failing mapping and valid-statistics tests**

```python
@pytest.mark.parametrize(("reward_type", "expected"), [
    ("gold_purchase", "strict_success"),
    ("valid_alternative_purchase", "near_miss"),
    ("partial_alternative_purchase", "near_miss"),
    ("wrong_purchase", "wrong_commit"),
    ("graceful_stop", "premature_stop"),
    ("early_abstain", "premature_stop"),
    ("repeat_loop", "exploration_stall"),
    ("max_steps", "exploration_stall"),
])
def test_failure_mapping(reward_type, expected):
    assert normalize_failure_type(valid_info(reward_type)).value == expected


def test_invalid_group_does_not_update_capability_statistics():
    state = CurriculumState.empty(manifest_hash="abc", config_hash="def", total_steps=10)
    state.update_group(invalid_group(), global_step=0)
    assert state.invalid_group_count == 1
    assert state.cells == {}


def test_zero_variance_patience_resets_on_variance_or_near_miss():
    state = state_fixture()
    for step in range(4):
        state.update_group(zero_variance_group(task_id=7), step)
    assert state.tasks[7].consecutive_zero_variance == 4
    state.update_group(varying_group(task_id=7), 4)
    assert state.tasks[7].consecutive_zero_variance == 0
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `pytest -q tests/test_curriculum_state.py`  
Expected: FAIL on imports.

- [ ] **Step 3: Implement the exact failure map and unknown-valid path**

Require `reward_valid is True` and a complete group before capability updates. Infrastructure/schema/reward invalid returns `invalid`. An unrecognized but valid terminal string returns `unknown_valid`, which contributes counts, utility, and variance but not failure-type or recoverability counts.

- [ ] **Step 4: Implement cumulative cells and task state**

Use dataclasses containing integer counts and float sums, not stored trajectories. Compute group population variance deterministically. Maintain cells keyed by `(difficulty_category, failure_type)`, category aggregates, invalid diagnostics, and task audit state. Success probability is `(strict_successes + 1)/(valid_trajectories + 2)`.

- [ ] **Step 5: Implement versioned checkpoint round-trip**

```python
def test_checkpoint_roundtrip_and_hash_guard():
    original = populated_state()
    restored = CurriculumState.from_checkpoint(
        original.to_checkpoint(),
        expected_manifest_hash=original.manifest_hash,
        expected_config_hash=original.config_hash,
        expected_total_steps=original.total_steps,
    )
    assert restored == original
    with pytest.raises(ValueError, match="manifest hash"):
        CurriculumState.from_checkpoint(
            original.to_checkpoint(), "different", original.config_hash, original.total_steps
        )
```

Reject unknown schema, phase inconsistency, config/manifest/total-step mismatch, negative counters, non-finite sums, and a stored phase inconsistent with committed step.

- [ ] **Step 6: Run and commit**

Run: `pytest -q tests/test_curriculum_state.py`  
Expected: PASS.

```bash
git add src/shopping_grpo/training/grpo/failure_types.py src/shopping_grpo/training/grpo/curriculum tests/test_curriculum_state.py
git commit -m "feat: track curriculum failure statistics"
```

---

### Task 7: Dynamic Priority and Deterministic Mixed Sampler

**Files:**

- Create: `src/shopping_grpo/training/grpo/curriculum/priority.py`
- Create: `src/shopping_grpo/training/grpo/curriculum/sampler.py`
- Create: `tests/test_curriculum_sampler.py`

**Interfaces:**

- Consumes: manifest tasks, `CurriculumState`, committed step, total steps, seed, batch size, generation-batch index.
- Produces: `phase_for_step(...)`, `compute_task_priority(...)`, and `CurriculumSampler.select(...) -> SelectionResult`.

- [ ] **Step 1: Write failing phase and priority tests**

```python
def test_switch_boundary_uses_committed_global_step():
    assert switch_step(10, 0.30) == 3
    assert phase_for_step(2, 10, 0.30) == "baseline_30"
    assert phase_for_step(3, 10, 0.30) == "difficulty_70"


def test_mixed_learnable_category_beats_stalled_hard_category():
    mixed = priority_fixture(category="medium", variance=.20, success=.5, near=.4, difficulty=.6)
    stalled = priority_fixture(category="hard", variance=0, success=0, near=0, stall=1, difficulty=1)
    assert compute_task_priority(mixed) > compute_task_priority(stalled)


def test_priority_formula_is_frozen():
    value = compute_learnability(v=.05, p=.5, near=.2, wrong=.1, stale_steps=25, difficulty=.8)
    expected = .30*.5 + .25*1 + .25*.275 + .10*1 + .10*.8
    assert value == pytest.approx(expected)
```

- [ ] **Step 2: Write failing sampling and recovery tests**

```python
def test_selection_is_unique_positive_and_reproducible():
    left = sampler_fixture(seed=9).select(global_step=4, generation_batch_index=0, count=3)
    right = sampler_fixture(seed=9).select(global_step=4, generation_batch_index=0, count=3)
    assert left.task_ids == right.task_ids
    assert len(set(left.task_ids)) == 3
    assert all(p > 0 for p in left.theoretical_probabilities.values())


def test_nonfinite_priority_falls_back_then_trips_patience():
    sampler = sampler_fixture(priority_override=float("nan"), fallback_patience=3)
    for step in range(2):
        assert sampler.select(step, 0, 2).fallback_reason == "non_finite_priority"
    with pytest.raises(RuntimeError, match="fallback patience"):
        sampler.select(2, 0, 2)
```

- [ ] **Step 3: Run tests and verify missing APIs**

Run: `pytest -q tests/test_curriculum_sampler.py`  
Expected: FAIL on imports.

- [ ] **Step 4: Implement the frozen v1 formula**

Implement exactly:

```python
variance_signal = v / (v + 0.05)
success_uncertainty = 4.0 * p * (1.0 - p)
recoverability = min(max(near_rate + 0.75 * wrong_rate, 0.0), 1.0)
staleness = min(max(steps_since_update / 25.0, 0.0), 1.0)
learnability = (
    0.30 * variance_signal
    + 0.25 * success_uncertainty
    + 0.25 * recoverability
    + 0.10 * staleness
    + 0.10 * probe_difficulty
)
```

Multiply by `1 - 0.75*p*p`, by a stall factor no lower than `0.35` when there is stall without near miss, and by `0.25` after four consecutive zero-variance groups. Clip the result to `[0.05,4.0]`.

- [ ] **Step 5: Implement deterministic weighted sampling without replacement**

Validate `switch_ratio in (0,1)`, `uniform_ratio in (0,1]`, sufficient unique legal tasks, and finite positive bounds. Derive a local RNG seed from SHA-256 of canonical JSON containing experiment seed, committed step, and generation-batch index; never consume module-global RNG. At each draw, calculate `0.20/N + 0.80*priority/sum(priority)` over remaining tasks, draw once, remove the task, and renormalize. Track actual uniform/dynamic branch counts by drawing the mixture branch first with the same local RNG.

- [ ] **Step 6: Preserve the baseline selector before the switch**

`CurriculumSampler.select` must accept a `baseline_select(count)` callback. For `global_step < S`, return exactly that callback result without inspecting priority state or consuming curriculum RNG. Add a test comparing the callback call/order byte-for-byte.

- [ ] **Step 7: Run and commit**

Run: `pytest -q tests/test_curriculum_sampler.py`  
Expected: PASS.

```bash
git add src/shopping_grpo/training/grpo/curriculum/priority.py src/shopping_grpo/training/grpo/curriculum/sampler.py tests/test_curriculum_sampler.py
git commit -m "feat: sample GRPO tasks by learnability"
```

---

### Task 8: Configuration, New Training Entry, and Preflight

**Files:**

- Modify: `configs/grpo.yaml`
- Create: `configs/grpo_evidence_curriculum.yaml`
- Create: `scripts/train_grpo_evidence_curriculum.py`
- Modify: `scripts/check_grpo_runtime.py`
- Create: `tests/test_evidence_curriculum_cli.py`
- Modify: `tests/test_grpo_dynamic_sampling_config.py`
- Modify: `tests/test_public_entrypoints.py`

**Interfaces:**

- Consumes: the existing `train_grpo.py` path validation/command builder plus manifest path.
- Produces: a dry-run-safe feature entry and `validate_curriculum_config(...)`/`validate_trajectory_credit_config(...)` preflight checks.

- [ ] **Step 1: Write failing baseline-default and new-entry tests**

```python
def test_baseline_config_keeps_both_features_off():
    cfg = yaml.safe_load(Path("configs/grpo.yaml").read_text())
    assert cfg["shopping_curriculum_sampling"]["enable"] is False
    assert cfg["shopping_trajectory_credit"]["enable"] is False


def test_curriculum_entry_enables_exact_features_and_switch(tmp_path):
    args = curriculum_args(tmp_path, dry_run=True)
    command, env = build_curriculum_command(args)
    assert "shopping_curriculum_sampling.enable=true" in command
    assert "shopping_curriculum_sampling.switch_ratio=0.30" in command
    assert "shopping_trajectory_credit.enable=true" in command
    assert not Path(env["GRPO_OUTPUT_DIR"]).exists()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_evidence_curriculum_cli.py tests/test_grpo_dynamic_sampling_config.py`  
Expected: FAIL because configs and entry are absent.

- [ ] **Step 3: Add default-off namespaces and explicit feature config**

Add the exact defaults from spec section 12 to `configs/grpo.yaml`. Build `grpo_evidence_curriculum.yaml` by Hydra composition of `grpo`, overriding only the two enable flags, `switch_ratio`, and manifest env interpolation. Do not duplicate the full GRPO recipe.

- [ ] **Step 4: Implement the entry by importing baseline helpers**

Refactor only reusable parsing/building helpers in `scripts/train_grpo.py` if required, without changing its parser or output. `train_grpo_evidence_curriculum.py` adds `--difficulty-manifest`, selects `grpo_evidence_curriculum`, injects explicit overrides, and otherwise uses the same model/data/environment/output/logger/preflight flow. Its dry-run prints the audit and returns before output directory creation or rollout.

- [ ] **Step 5: Extend preflight with strict validation**

Validate manifest existence/schema/hash, exact train coverage, split/Evaluation isolation, `(0,1)` switch ratio, `(0,1]` uniform ratio, finite ordered priority bounds, patience values, evidence version, scores, multiplier bounds, tolerance, fallback rate/window, and checkpoint compatibility when resuming. Keep baseline preflight unchanged when both switches are false.

- [ ] **Step 6: Run CLI/config tests and baseline command regression**

Run: `pytest -q tests/test_evidence_curriculum_cli.py tests/test_grpo_dynamic_sampling_config.py tests/test_public_entrypoints.py tests/test_verl_compat.py`  
Expected: PASS.

- [ ] **Step 7: Commit entry and configuration**

```bash
git add configs/grpo.yaml configs/grpo_evidence_curriculum.yaml scripts/train_grpo.py scripts/train_grpo_evidence_curriculum.py scripts/check_grpo_runtime.py tests/test_evidence_curriculum_cli.py tests/test_grpo_dynamic_sampling_config.py tests/test_public_entrypoints.py
git commit -m "feat: add evidence curriculum GRPO entry"
```

---

### Task 9: AgentLoop Evidence and Stable Turn Metadata

**Files:**

- Modify: `src/shopping_grpo/training/grpo/adapter/runtime.py`
- Modify: `src/shopping_grpo/training/grpo/adapter/tools.py`
- Modify: `src/shopping_grpo/training/grpo/adapter/agent_loop.py`
- Modify: `tests/test_verl_adapter.py`
- Create: `tests/test_agent_loop_credit_metadata.py`

**Interfaces:**

- Consumes: tool-call lifecycle and veRL AgentLoop generated response boundaries.
- Produces: `shopping.evidence_events`, `shopping.assistant_turn_spans`, and alignment/fallback metadata.

- [ ] **Step 1: Write failing tool-lifecycle evidence tests**

```python
async def test_tool_records_evidence_before_projection():
    output = await run_one_tool_with_raw_state_change("search", before_ids={"a"}, after_ids={"a", "b"})
    event = output.extra_fields["shopping"]["evidence_events"][0]
    assert event["new_search_result_set"] is True
    assert event["new_product_count"] == 1


async def test_guard_rejection_gets_same_turn_id_and_negative_event():
    output = await run_guard_rejected_tool(turn_id=3)
    event = output.extra_fields["shopping"]["evidence_events"][0]
    assert event["assistant_turn_id"] == 3
    assert event["guard_rejection"] is True
```

- [ ] **Step 2: Write failing turn-span and compaction tests**

Assert every Assistant generation, including `think`, has one event and one ordered half-open response-token span. Assert `context_compactions > 0` exports `credit_alignment_valid=false` and reason `context_compaction_alignment_unsupported`.

- [ ] **Step 3: Run tests and verify metadata is absent**

Run: `pytest -q tests/test_agent_loop_credit_metadata.py tests/test_verl_adapter.py`  
Expected: FAIL on missing fields.

- [ ] **Step 4: Extend trajectory-local runtime state**

Add `next_assistant_turn_id`, `active_assistant_turn_id`, `evidence_events`, `assistant_turn_spans`, transient public snapshot, and `credit_alignment_valid/reason`. Do not place target/gold fields into state. Export only validated event dictionaries and integer spans.

- [ ] **Step 5: Record raw-state evidence at the tool boundary**

Capture the public snapshot before Guard/environment execution and after the raw environment result is available, before `project_observation`. For Guard rejection, exact repeat, terminal, and `think`, create the corresponding event even if no environment step is executed. Remove the current exported full `actions` list from the new credit diagnostic path; retain existing baseline diagnostics only if required by current tests, and never copy it into new aggregate logs.

- [ ] **Step 6: Capture spans from actual generated token indices**

At the AgentLoop generation boundary, record the response-array length before and after appending each Assistant generation. Associate that half-open interval with the active turn ID. Validate spans against final response IDs before exporting. If veRL compaction has occurred or validation fails, export the complete events/spans for diagnosis but set alignment invalid so weighting falls back.

- [ ] **Step 7: Run adapter and context tests**

Run: `pytest -q tests/test_agent_loop_credit_metadata.py tests/test_verl_adapter.py tests/test_context_window.py tests/test_observation_projection.py`  
Expected: PASS.

- [ ] **Step 8: Commit AgentLoop metadata**

```bash
git add src/shopping_grpo/training/grpo/adapter/runtime.py src/shopping_grpo/training/grpo/adapter/tools.py src/shopping_grpo/training/grpo/adapter/agent_loop.py tests/test_verl_adapter.py tests/test_agent_loop_credit_metadata.py
git commit -m "feat: attach evidence and turn spans to rollouts"
```

---

### Task 10: Thin veRL Hooks, Checkpoint State, and Diagnostics

**Files:**

- Modify: `patches/verl-0.8.0-shopping-dynamic-sampling.patch`
- Modify: `scripts/apply_verl_dynamic_sampling_patch.py`
- Modify: `scripts/check_grpo_runtime.py`
- Modify: `src/shopping_grpo/training/grpo/dynamic_sampling.py`
- Modify: `tests/test_verl_dynamic_sampling.py`
- Modify: `tests/test_verl_dynamic_sampling_patch.py`
- Create: `tests/test_verl_curriculum_contract.py`

**Interfaces:**

- Consumes: project sampler/credit APIs and veRL batch/checkpoint boundaries.
- Produces: selected task batches, weighted token advantages, committed sampler checkpoint, and aggregate JSON diagnostics.

- [ ] **Step 1: Write failing static patch-contract tests**

```python
def test_patch_has_only_thin_project_calls():
    patch = PATCH.read_text()
    for symbol in (
        "CurriculumSampler", "update_group", "project_token_advantages",
        "to_checkpoint", "from_checkpoint",
    ):
        assert symbol in patch
    assert "0.30 * variance_signal" not in patch
    assert "near_miss_rate +" not in patch


def test_existing_group_filter_remains_a_separate_call():
    patch = PATCH.read_text()
    assert "select_reward_varying_groups(" in patch
    assert patch.index("select_reward_varying_groups(") < patch.index("project_token_advantages(")
```

- [ ] **Step 2: Write failing synthetic trainer-boundary tests**

Use lightweight fake batches to verify: pre-switch delegates to baseline order; post-switch requests unique tasks; each selected uid still has exactly `rollout.n`; invalid/constant groups are filtered before statistics and credit; zero advantage remains zero; disabled credit returns byte/elements-equivalent advantages; skipped updates do not advance committed step.

- [ ] **Step 3: Run contract tests and verify failure**

Run: `pytest -q tests/test_verl_curriculum_contract.py tests/test_verl_dynamic_sampling_patch.py`  
Expected: FAIL because hooks/marker are absent.

- [ ] **Step 4: Add task selection and group update hooks**

Before requesting the next generation batch, call `CurriculumSampler.select` only when enabled and at/after `S`; feed selected task IDs through a minimal task-id lookup/batch-index interface instead of rebuilding the dataset. During baseline phase, execute the original iterator expression unchanged. After Reward extraction and complete-group validation, pass one immutable group summary to `CurriculumState.update_group`; keep `select_reward_varying_groups` authoritative for accepted training groups and the three-batch cap.

- [ ] **Step 5: Add post-advantage credit hook**

Immediately after standard GRPO advantage computation and before actor update, iterate trajectories, validate `shopping` evidence/spans, and call `project_token_advantages`. Replace only the advantage tensor values at response-mask positions. Do not touch rewards, uid grouping, old log probabilities, reference log probabilities, KL fields, or masks. When disabled or on fallback, preserve the original tensor exactly.

- [ ] **Step 6: Add committed checkpoint hooks**

Store sampler state under a namespaced `shopping_curriculum_state.json` alongside each successful veRL checkpoint. Save only after optimizer update and committed step advance. On resume, validate and restore it before selecting tasks. Require the sidecar when curriculum is enabled and a checkpoint is resumed; reject missing, mismatched, or future-schema state.

- [ ] **Step 7: Add aggregate safe diagnostics**

Extend `append_training_diagnostic` calls with stage, switch step, category counts/probabilities, uniform/dynamic counts, priority summary, failure proportions, fallback counts/reasons, unique-task rate, valid-group yield, skipped updates, evidence proportions, multiplier summary, and mass error. Explicitly filter payload keys so ASIN, raw Observation, action parameters, gold, and hidden fields cannot serialize.

- [ ] **Step 8: Regenerate and verify the pinned patch**

Apply the edited patch to a verified pristine veRL 0.8.0 `ray_trainer.py`, compile it, compute SHA-256, bump marker to `SHOPPING_GRPO_EVIDENCE_CURRICULUM_PATCH_V1`, and set the new expected patched hash in `apply_verl_dynamic_sampling_patch.py` and preflight. Keep the verified original hash and rollback behavior.

- [ ] **Step 9: Run all patch and synthetic integration tests**

Run: `pytest -q tests/test_verl_dynamic_sampling.py tests/test_verl_dynamic_sampling_patch.py tests/test_verl_curriculum_contract.py tests/test_verl_compat.py`  
Expected: PASS.

- [ ] **Step 10: Commit the integration**

```bash
git add patches/verl-0.8.0-shopping-dynamic-sampling.patch scripts/apply_verl_dynamic_sampling_patch.py scripts/check_grpo_runtime.py src/shopping_grpo/training/grpo/dynamic_sampling.py tests/test_verl_dynamic_sampling.py tests/test_verl_dynamic_sampling_patch.py tests/test_verl_curriculum_contract.py tests/test_verl_compat.py
git commit -m "feat: integrate evidence curriculum with veRL"
```

---

### Task 11: End-to-End Contracts, Documentation, and Final Verification

**Files:**

- Create: `tests/test_evidence_curriculum_acceptance.py`
- Modify: `docs/grpo.md`
- Modify: `README.md`
- Modify: `README.en.md`

**Interfaces:**

- Consumes: all completed feature interfaces.
- Produces: synthetic acceptance coverage and operator documentation.

- [ ] **Step 1: Add a synthetic end-to-end acceptance test**

Build a six-task in-memory manifest covering easy/medium/hard, feed baseline statistics through step `S-1`, verify the switch at `S`, update near-miss/varying and hard/stalled groups, checkpoint, restore, and assert identical subsequent task IDs. Feed a fixed reward-varying batch through raw GRPO advantage then responsibility projection and assert direction, bounds, mass preservation, and group completeness.

```python
def test_single_run_switch_credit_and_resume_contract():
    run = SyntheticCurriculumRun(total_steps=10, switch_ratio=.30, seed=17)
    assert [run.phase_at(i) for i in (0, 2, 3, 9)] == [
        "baseline_30", "baseline_30", "difficulty_70", "difficulty_70"
    ]
    run.observe(group_near_miss_varying("medium"), step=2)
    run.observe(group_all_stall_zero_variance("hard"), step=2)
    selected = run.select(step=3)
    restored = SyntheticCurriculumRun.restore(run.checkpoint())
    assert restored.select(step=4).task_ids == run.select(step=4).task_ids
    assert len(set(selected.task_ids)) == len(selected.task_ids)
```

- [ ] **Step 2: Add disabled-feature equivalence assertions**

Compare the existing baseline command/audit before and after config additions, fixed callback task order, raw advantage tensors, actor-loss inputs, and response masks. The test must use exact equality, not approximate equality, when both new switches are false.

- [ ] **Step 3: Document operator workflow**

Update `docs/grpo.md` with exact commands for:

```bash
python scripts/probe_grpo_difficulty.py --dry-run --model outputs/models/sft-merged --output outputs/probes/fresh-sft-v1
python scripts/probe_grpo_difficulty.py --model outputs/models/sft-merged --output outputs/probes/fresh-sft-v1
python scripts/train_grpo_evidence_curriculum.py --dry-run --difficulty-manifest outputs/probes/fresh-sft-v1/difficulty_manifest.jsonl
```

Explain that the second command executes Probe inference and the training command without `--dry-run` must only be run on explicit user request. Document artifacts, hashes, invalid-task isolation, both ablation switches, stage semantics, fallback/abort diagnostics, and checkpoint resume rejection rules. Add brief links from both READMEs without expanding the supported workflow.

- [ ] **Step 4: Run focused acceptance and dry-run tests**

Run: `pytest -q tests/test_evidence_curriculum_acceptance.py tests/test_evidence_curriculum_cli.py tests/test_probe_grpo_difficulty.py`  
Expected: PASS; commands do not create output directories under dry-run.

- [ ] **Step 5: Run the complete non-training verification suite**

Run: `pytest -q`  
Expected: all tests PASS.

Run: `.venv/bin/ruff check src scripts tests`  
Expected: PASS.

Run: `.venv/bin/mypy src/shopping_grpo`  
Expected: PASS, or only pre-existing errors documented with an unchanged baseline comparison.

Run: `git diff --check`  
Expected: no whitespace errors.

- [ ] **Step 6: Audit forbidden changes and data overlap without running training**

Run:

```bash
git diff --name-only HEAD~10..HEAD
grep -RIn --exclude-dir=.git -E 'target_asin|hidden_constraints|gold_' src/shopping_grpo/training/grpo/evidence.py src/shopping_grpo/training/grpo/trajectory_credit.py
python scripts/probe_grpo_difficulty.py --dry-run --model outputs/models/sft-merged --output outputs/probes/fresh-sft-v1
python scripts/train_grpo_evidence_curriculum.py --dry-run --difficulty-manifest tests/fixtures/difficulty_manifest.jsonl
```

Expected: only scoped files changed; the grep finds no serialized hidden fields; both dry-runs report valid audits and create no output directories. Do not remove any pre-existing output directory to make this assertion pass—use a fresh temporary path in the actual test.

- [ ] **Step 7: Commit tests and documentation**

```bash
git add tests/test_evidence_curriculum_acceptance.py docs/grpo.md README.md README.en.md
git commit -m "docs: document evidence curriculum GRPO"
```

---

## Completion Criteria

- Every task's focused tests pass before its commit.
- The full non-training suite and static checks pass at the final revision.
- Original `scripts/train_grpo.py` behavior and disabled-feature tensors are exactly equivalent.
- No task ID overlaps Evaluation, and invalid Probe tasks cannot enter training.
- Responsibility multipliers remain in `[0.5,1.5]`, preserve token-weighted mean `1±1e-6`, and fall back for the whole trajectory on any alignment uncertainty.
- Step `S-1` uses baseline selection and step `S` uses curriculum selection, including after restore.
- The task sequence after restore is deterministic for the same data, manifest, config, seed, and checkpoint.
- No GRPO training, model merge, or 200-task Evaluation was launched during implementation verification.
