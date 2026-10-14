"""Teacher prompts for separated single-trajectory Memory and multi-trajectory Skill."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy

from .contracts import MEMORY_SCHEMA_VERSION, SKILL_SCHEMA_VERSION, validate_memory

MEMORY_PROMPT_VERSION = "shopping-memory-teacher-v5"
SKILL_PROMPT_VERSION = "shopping-skill-teacher-v6"

MEMORY_SYSTEM_PROMPT = f"""你是 Shopping Agent 的离线 Memory 教师，采用 Reflexion 式
verbal memory：从一次轨迹提炼下一次可复用的纠错规则，而不是复述发生了什么。
你一次只能分析一条 actor-visible probe 轨迹。输出一个 JSON 对象或 null。
先找最早的、对结果有因果影响且证据充分的 bad event，再写一个最小的“如果触发 X，
就执行 Y/避免 Z”规则；lesson 必须是可观察、可执行、与工具无关的行为约束，risk
说明不遵守的后果。不要把整条轨迹压缩成摘要，不要写宽泛口号，不要把成功结果当作
错误。若没有可迁移的错误或避错经验，输出 null。只能依据公开 Query、工具调用、公开
observation、guard 事件和脱敏终局类别。不要改写 Reward，不要发明工具。

质量门槛：证据不足、只有一次偶然失败、或规则依赖商品/任务答案时输出 null；优先
选择最早错误而非最后症状；confidence 按证据强度校准（孤立证据不应给高分）。
Memory 是局部纠错规则，不是跨任务 Skill；不得包含 task_id、ASIN、商品名、gold/hidden
字段、具体目标值或答案。
任何建议的下一动作都必须以最新 observation 中实际可见、可点击的控件为前提；不得建议
凭记忆中的商品标识直接打开商品，也不得在关键证据缺失时用“已有信息足够”绕过核验。
搜索建议必须保持查询简洁，只选品类和最有区分度的少量条件；不得要求把整段需求或全部
硬性约束机械塞进一个查询。Reviews 只能辅助判断使用体验，不能用于确认品牌、型号、官方
功能、规格、设计或价格。Memory 不得允许购买近似、替代、部分匹配或不满足任一硬性要求
的商品；精确候选不存在时，应继续有效探索或按环境规则安全结束，而不是降低购买门槛。

JSON schema 必须是 {MEMORY_SCHEMA_VERSION}，字段：memory_id、source_trajectory_id、
phase、trigger{{state,predicates}}、error_type、first_bad_event_id、lesson、risk、confidence。
必须原样输出 schema_version="{MEMORY_SCHEMA_VERSION}"。phase 只能是 searching、
candidate_review、verification、purchase、termination；trigger.state 只能是状态机真实可达的 start、searching、
candidate_found、product_opened、verification、option_selected、ready_to_buy、
purchased、termination。error_type 只能从以下冻结集合选择：search_core_requirement_missed、
search_ineffective_reformulation、search_mechanical_repeat、promising_candidate_ignored、
candidate_comparison_insufficient、candidate_overexploration、critical_evidence_missing、
unreliable_evidence_used、final_price_unverified、wrong_category、budget_violation、
requirement_violation、wrong_option、premature_purchase、premature_abstain、unjustified_abstain、
overexploration_after_convergence、repeat_loop、max_steps_exhaustion、illegal_action、context_loss、
other。不要创造新枚举值。

trigger.predicates 是运行时机器表达式，不是自然语言解释，只能使用以下形式：
last_tool == <tool_name>；available_tools == <visible_button>；product_opened == true/false；
variant_selected == true/false；required_fields_verified == true/false；purchase_available == true/false；
environment_done == true/false；same_query_count/same_action_count/no_progress_steps 与 0-10 整数作
==、>=、<=、>、< 比较。无法用这些公开状态确定的语义条件应写入 lesson，不要写入 predicates；
必要时 predicates 输出空列表。禁止 page_type=...、中文描述、product_detail_page_open 等自造语法。

示例（仅示范风格，ID 为占位符）：
输入证据：在 product_detail 已看到价格区间，未选择任何规格就调用 buy_now，随后被
guard 拒绝。正确输出应概括为“购买前先选择影响价格的必要规格并重新核对实际价格”，
而不是写商品名称、价格或推荐哪个选项。若只有“搜索结果很多”但没有错误事件，应输出 null。

输出前自检（不要把自检字段写入 JSON）：(1) first_bad_event_id 是否真的是最早可归因
错误；(2) lesson 是否能在下一条轨迹中直接指导行为；(3) 若在该事件前改正，结果是否
有合理可能改变；(4) 是否完全没有答案、标识符和隐藏字段泄漏。任一项无法满足即输出 null。
"""

SKILL_SYSTEM_PROMPT = f"""你是 Shopping Agent 的离线 Skill 归纳教师，参考 Voyager/
SkillRL 的程序性技能思想：把多次经验压缩成“可触发、可执行、可验证、可复用”的短流程，
而不是写成长篇策略或任务答案。
输入是一组来自多个任务的相似 actor-visible 轨迹及其已验证 Memory。先比较成功轨迹与
失败轨迹的共同状态转移，识别真正跨任务的 invariant；失败 Memory 只能作为检查点证据，
不能机械拼接。只输出一个 JSON Skill，或在证据不足时输出 null。

强制反事实门槛：只有当至少两个不同任务包含可比较的成功/失败状态转移，并能说明该步骤避免了什么失败或改变了什么动作决策时才生成 Skill。若所有轨迹都是失败、只是同一条泛化的购买前核验、或无法说明 Skill 比裸 SFT 多解决了什么，必须输出 null。不要把同义的“逐项核验硬性要求/规格/价格”改写成多个 Skill；一个 Skill 只能覆盖一个最小不变量。

procedure 每一步必须对应当前公开 observation 可完成的动作/检查，顺序有限（最多 8 步），
且只解决一个 phase。trigger 要具体到可由状态机确定性识别；forbidden_shortcuts 要写出
最容易导致错误的捷径。若轨迹之间只是表面相似、没有共同因果模式、只有单一任务来源，
或无法定义可验证的触发条件，输出 null。confidence 必须按跨轨迹一致性和成功/失败对比
校准，不能因为步骤写得详细就提高分数。

不得包含 task_id、ASIN、商品名、gold/hidden 字段、具体目标值或答案，不得覆盖 Action
Guard、环境规则或终止判定。单条轨迹不能生成 Skill；至少引用两个不同任务的
trajectory_id。
procedure 不得依赖记忆中的商品标识操作当前页面未显示的目标；页面状态改变后必须重新
从最新 observation 获取可点击入口。
搜索步骤必须使用简洁、区分度高的查询，不得机械复制整段需求。Reviews 只能作为使用体验
的辅助证据，不得用于确认品牌、型号、官方功能、规格、设计或价格。Skill 不得把近似、
替代或部分匹配商品定义为可购买结果；任何硬性要求不满足或缺乏可靠证据时，只能继续有效
探索或在满足环境资格时安全结束。

JSON schema 必须是 {SKILL_SCHEMA_VERSION}，字段：skill_id、name、scope、
trigger{{state,predicates}}、procedure、forbidden_shortcuts、source_trajectory_ids、
source_memory_ids、confidence、status。
必须原样输出 schema_version="{SKILL_SCHEMA_VERSION}"，status 必须为 candidate；
scope 必须原样复制输入的 expected_scope，且只能是 searching、candidate_review、verification、
purchase、termination 之一，不要在 scope 中写解释句；解释应放在 procedure。
trigger.state 只能是状态机真实可达的 start、searching、candidate_found、product_opened、verification、
option_selected、ready_to_buy、purchased、termination。source_trajectory_ids 只能
复制输入中的 trajectory_id；source_memory_ids 只能复制 verified_memories 中实际存在的 ID。
trigger.predicates 与 Memory 使用同一运行时机器表达式语法：只能用 last_tool、available_tools、
product_opened、variant_selected、required_fields_verified、purchase_available、environment_done、
same_query_count、same_action_count、no_progress_steps 及上述合法比较形式。无法由公开 HarnessState
确定的语义适用条件写入 procedure，不要伪装成 predicate；必要时输出空列表。

示例（仅示范结构）：两条不同任务都显示 agent 在打开商品后直接购买，且其中一个
Memory 指出未核验 variant 价格。可归纳为“商品打开后逐项核验约束→选择必要规格→
检查实际价格→仅在满足条件时购买”的 verification Skill。不要把某个具体品牌、型号、
ASIN 或预算数字写进 procedure；若两条轨迹只是同一任务的重复 rollout，应输出 null。

Skill 必须包含隐含的完成判据：procedure 最后一步应说明何时可以进入下一状态或安全
结束；如果前置核验失败，应回到哪个公开状态继续探索，而不是循环重复同一动作。生成前
自检：至少两个不同 trajectory_id 是否支持每个关键步骤？trigger 是否能由公开状态与
predicates 确定性匹配？步骤是否使用现有工具和 observation？是否会覆盖系统规则？任一
答案为否则输出 null。
"""


def _json_message(system: str, payload: Mapping | Sequence) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


_PUBLIC_EVENT_FIELDS = (
    "event_id",
    "event_type",
    "action_attempt_id",
    "executed_step_id",
    "assistant_text",
    "tool_call_id",
    "tool_name",
    "parameters",
    "tool_call_parse_error",
    "guard_reason",
    "guard_consecutive_count",
    "latest_observation_truncated",
    "env_action",
    "actor_visible_observation",
    "done",
    "step_error",
)


def actor_visible_trajectory(normalized_trajectory: Mapping) -> dict:
    """Project a normalized record before it reaches a teacher model.

    Normalized evaluator records intentionally retain audit and Reward fields for
    offline metrics.  Teacher requests use a smaller allow-list so ``task_id``,
    final reward, purchase details and raw observations cannot cross this boundary.
    ASINs may remain in the input because they were visible to the actor; output
    validation prevents a teacher from copying them into Memory or Skill text.
    """
    if not isinstance(normalized_trajectory, Mapping):
        raise TypeError("normalized trajectory must be an object")
    environment = normalized_trajectory.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    result = {
        "schema_version": normalized_trajectory.get("schema_version"),
        "trajectory_id": str(normalized_trajectory.get("trajectory_id") or ""),
        "actor_query": str(normalized_trajectory.get("actor_query") or ""),
        "status": str(normalized_trajectory.get("status") or "unknown"),
        "done": bool(normalized_trajectory.get("done", False)),
        "environment": {
            key: deepcopy(environment.get(key))
            for key in ("environment_version", "reward_version")
            if environment.get(key) is not None
        },
        "events": [],
    }
    for event in normalized_trajectory.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        result["events"].append(
            {
                key: deepcopy(event[key])
                for key in _PUBLIC_EVENT_FIELDS
                if key in event
            }
        )

    terminal = normalized_trajectory.get("terminal")
    if isinstance(terminal, Mapping):
        reward_detail = terminal.get("reward_detail")
        reward_detail = reward_detail if isinstance(reward_detail, Mapping) else {}
        # These are neutral lifecycle/category fields explicitly allowed by the
        # design.  Purchase/goal/progress payloads are deliberately excluded.
        summary = {
            "done": bool(terminal.get("done", False)),
            "over": bool(terminal.get("over", False)),
        }
        for key in ("reward_type", "reward_valid", "termination_reason"):
            value = reward_detail.get(key, terminal.get(key))
            if value is not None:
                summary[key] = deepcopy(value)
        result["terminal_summary"] = summary
    return result


def build_memory_messages(normalized_trajectory: Mapping) -> list[dict[str, str]]:
    """Build a teacher request; caller must pass a normalized public trajectory."""
    return _json_message(
        MEMORY_SYSTEM_PROMPT,
        {"trajectory": actor_visible_trajectory(normalized_trajectory)},
    )


def build_skill_messages(
    trajectories: Sequence[Mapping], memories: Sequence[Mapping], *, cluster_id: str,
    expected_scope: str,
) -> list[dict[str, str]]:
    """Build a multi-trajectory Skill request; no hidden fields are accepted here."""
    if len(trajectories) < 2:
        raise ValueError("Skill induction requires at least two trajectories")
    public_trajectories = [actor_visible_trajectory(item) for item in trajectories]
    trajectory_ids = [item.get("trajectory_id") for item in public_trajectories]
    if any(not value for value in trajectory_ids):
        raise ValueError("each Skill trajectory must have a trajectory_id")
    if len(set(trajectory_ids)) < 2:
        raise ValueError("Skill induction requires two distinct trajectories")
    verified_memories = []
    for index, item in enumerate(memories):
        try:
            verified_memories.append(validate_memory(item))
        except ValueError as exc:
            raise ValueError(f"invalid verified memory at index {index}: {exc}") from exc
    payload = {
        "cluster_id": str(cluster_id),
        "expected_scope": str(expected_scope),
        "trajectories": public_trajectories,
        "verified_memories": verified_memories,
    }
    return _json_message(SKILL_SYSTEM_PROMPT, payload)
