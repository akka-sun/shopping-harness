#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.summary import summarize_trajectories
from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--infrastructure-retries", type=int, default=3,
        help="每个任务遇到环境/服务基础设施错误时的有限重试次数",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=24576,
        help="上下文窗口；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument(
        "--observation-token-budget",
        type=int,
        default=1536,
        help="Observation token 预算；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument(
        "--harness-memories",
        type=Path,
        help="validated Memory JSONL to inject as ephemeral runtime guidance",
    )
    parser.add_argument(
        "--harness-skills",
        type=Path,
        help="validated Skill JSONL to inject as ephemeral runtime guidance",
    )
    parser.add_argument(
        "--activate-candidate-skills",
        action="store_true",
        help="activate candidate Skills in memory for this qualification run only",
    )
    parser.add_argument("--harness-source-trajectories", type=Path)
    parser.add_argument("--harness-semantic-threshold", type=float, default=0.075)
    parser.add_argument("--harness-task-threshold", type=float, default=0.08)
    parser.add_argument("--harness-skill-threshold", type=float, default=0.10)
    parser.add_argument("--harness-skill-task-threshold", type=float, default=0.10)
    parser.add_argument("--harness-product-weight", type=float, default=0.25)
    parser.add_argument("--harness-resolved-threshold", type=float, default=0.26)
    parser.add_argument(
        "--harness-blocked-skill-id", action="append", default=[],
        help="skill id quarantined from runtime retrieval; may be repeated",
    )
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.infrastructure_retries < 0:
        raise SystemExit("--infrastructure-retries 不能为负数")
    if args.context_window < 0:
        raise SystemExit("--context-window 不能为负数")
    if args.context_window and args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    if args.observation_token_budget < 0:
        raise SystemExit("--observation-token-budget 不能为负数")
    for name, value in (
        ("--harness-semantic-threshold", args.harness_semantic_threshold),
        ("--harness-task-threshold", args.harness_task_threshold),
        ("--harness-product-weight", args.harness_product_weight),
        ("--harness-skill-threshold", args.harness_skill_threshold),
        ("--harness-skill-task-threshold", args.harness_skill_task_threshold),
        ("--harness-resolved-threshold", args.harness_resolved_threshold),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} 必须在 0 到 1 之间")
    if (args.harness_memories or args.harness_skills) and not args.harness_source_trajectories:
        raise SystemExit("启用 Harness 时必须提供 --harness-source-trajectories 以执行语义门控")
    tasks = load_tasks(args.benchmark)
    if args.task_start < 0 or (args.task_end is not None and args.task_end < args.task_start):
        raise SystemExit("任务切片范围无效")
    if args.task_id is not None:
        wanted = set(args.task_id)
        tasks = [task for task in tasks if int(task["task_id"]) in wanted]
    else:
        tasks = tasks[args.task_start:args.task_end]
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )
    harness_bank = None
    if args.harness_memories or args.harness_skills:
        from Harness.runtime import KnowledgeBank

        memories = _read_jsonl(args.harness_memories) if args.harness_memories else []
        skills = _read_jsonl(args.harness_skills) if args.harness_skills else []
        if args.activate_candidate_skills:
            skills = [
                {**item, "status": "active"}
                if item.get("status") == "candidate"
                else item
                for item in skills
            ]
        source_map = {}
        if args.harness_source_trajectories:
            from Harness.semantic import source_queries
            source_map = source_queries(args.harness_source_trajectories)
        harness_bank = KnowledgeBank(memories, skills, source_queries=source_map)
    harness_semantic_gate = None
    if harness_bank is not None:
        from Harness.semantic import SemanticThresholdGate
        harness_semantic_gate = SemanticThresholdGate(
            threshold=args.harness_semantic_threshold,
            task_threshold=args.harness_task_threshold,
            skill_threshold=args.harness_skill_threshold,
            skill_task_threshold=args.harness_skill_task_threshold,
            product_weight=args.harness_product_weight,
            resolved_threshold=args.harness_resolved_threshold,
            blocked_skill_ids=args.harness_blocked_skill_id,
        )
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        harness_bank=harness_bank,
        harness_semantic_gate=harness_semantic_gate,
        infrastructure_retries=args.infrastructure_retries,
    )
    summary = summarize_trajectories(
        [task["task_id"] for task in tasks], _read_jsonl(args.output)
    )
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": args.model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "infrastructure_retries": args.infrastructure_retries,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
        "harness_semantic_threshold": args.harness_semantic_threshold,
        "harness_task_threshold": (
            args.harness_task_threshold
            if args.harness_task_threshold is not None
            else args.harness_semantic_threshold
        ),
        "harness_product_weight": args.harness_product_weight,
        "harness_skill_threshold": args.harness_skill_threshold,
        "harness_skill_task_threshold": args.harness_skill_task_threshold,
        "harness_resolved_threshold": args.harness_resolved_threshold,
        "harness_memories": str(args.harness_memories) if args.harness_memories else None,
        "harness_skills": str(args.harness_skills) if args.harness_skills else None,
        "activate_candidate_skills": bool(args.activate_candidate_skills),
        "harness_source_trajectories": (
            str(args.harness_source_trajectories)
            if args.harness_source_trajectories else None
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
