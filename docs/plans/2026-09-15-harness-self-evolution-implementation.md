# Harness 自进化实施文档

本文把 [Harness 自进化设计](2026-09-14-harness-self-evolution-design.md) 映射为当前
仓库中的可执行实现。目标是用确定性 SFT Probe 轨迹提取教师 Memory/Skill，并在运行时
按公开状态检索注入；不做第二轮 SFT，不执行 GRPO，不修改模型参数。

## 1. 已实现边界

代码位于仓库根目录 `Harness/`，与现有 GRPO/evaluation rollout 解耦：

| 文件 | 责任 |
| --- | --- |
| `contracts.py` | Memory/Skill schema、字段白名单、frozen error taxonomy、来源事件和泄漏校验 |
| `prompts.py` | 单轨迹 Memory 与多轨迹 Skill 的独立教师请求；normalized 轨迹 public projection |
| `pipeline.py` | JSONL dry-run 请求生成、artifact accepted/rejected 校验、原子写入 |
| `runtime.py` | trajectory-local 状态机、声明式 trigger matcher、单 Skill/有限 Memory 选择 |
| `integration.py` | 单回合临时 guidance 注入，不污染 canonical history |

`Harness/__main__.py` 提供 `python -m Harness` 入口。模块只读轨迹或知识 artifact，不
启动教师模型、ShopSimulator 或训练进程。

## 2. Probe 输入

`ProbeOutcome.trajectory` 保存完整 collector trajectory。运行
`scripts/probe_grpo_difficulty.py --raw-run-dir <dir>` 时，probe 会 append：

```text
<dir>/trajectories.jsonl   # 原始 collector 轨迹，仅供本地审计
<dir>/normalized.jsonl     # normalize_trajectory() 结果，供 Harness 教师输入
```

`probe_metadata.json` 记录 SFT 模型、train/validation/evaluation task 文件、轨迹文件的
SHA-256，以及 Environment v2.1、Reward v3、observation v2 和 tool schema v2 版本。Probe
仍强制 train/validation 与 `data/evaluation/tasks.jsonl` 零重叠。

## 3. 离线教师请求

Memory 严格逐轨迹生成：

```bash
PYTHONPATH=src python -m Harness memory-requests \
  --input outputs/harness/<run>/probe/normalized.jsonl \
  --output outputs/harness/<run>/memory/teacher_requests.jsonl
```

Skill 使用显式 cluster，且每个 cluster 至少两个不同 trajectory：

```json
{"cluster_id":"verification-1","trajectory_ids":["traj-a","traj-b"],"memory_ids":[]}
```

```bash
PYTHONPATH=src python -m Harness skill-requests \
  --trajectories outputs/harness/<run>/probe/normalized.jsonl \
  --clusters outputs/harness/<run>/skill/clusters.jsonl \
  --memories outputs/harness/<run>/memory/active.jsonl \
  --output outputs/harness/<run>/skill/teacher_requests.jsonl
```

这两个命令只写 OpenAI-compatible `messages` JSONL，不发请求。Memory/Skill 的教师输出
分别落盘后再校验：

```bash
PYTHONPATH=src python -m Harness validate --kind memory \
  --input outputs/harness/<run>/memory/candidates.jsonl \
  --trajectories outputs/harness/<run>/probe/normalized.jsonl \
  --accepted outputs/harness/<run>/memory/active.jsonl \
  --rejected outputs/harness/<run>/memory/rejected.jsonl
```

Skill 校验使用 `--kind skill`；不传 `--trajectories` 时只执行 schema/泄漏校验。已有
artifact 默认不可覆盖，显式传 `--force` 才替换。

## 4. 运行时接入

每次模型请求前由调用方更新状态并检索：

```python
from Harness import HarnessState, KnowledgeBank, complete_with_guidance

state = HarnessState()
state.update(tool_name=name, parameters=args, observation=visible_observation, done=done)
response, guidance = complete_with_guidance(
    client, canonical_messages, tool_schemas, bank, state, memory_limit=2
)
```

`complete_with_guidance` 只给本次 `client.complete` 使用附加 user message；不要把该消息
写回 canonical history。Harness 最多选一个 active Skill 和两条匹配 Memory；Action Guard、
环境规则和 Reward/终止判定仍由现有系统负责。当前实现不直接修改
`src/shopping_grpo/evaluation/rollout.py`，避免在 A/B 验证前改变默认评测行为。

## 5. 验证与晋升

已提供 `tests/test_harness.py`，覆盖 schema、泄漏拒绝、单/多轨迹约束、状态 trigger、
循环计数和临时注入；`tests/test_probe_grpo_difficulty.py` 额外覆盖 probe 轨迹落盘。
当前环境未安装 pytest/ruff，因此使用以下检查：

```bash
PYTHONPATH=src python -m compileall -q Harness scripts/probe_grpo_difficulty.py src tests
PYTHONPATH=src python -m Harness --help
```

教师输出进入 active bank 前仍需 holdout probe 配对验证：strict `gold_purchase` 不下降，
目标错误率下降，且无答案泄漏、非法动作、重复循环或上下文超限。完成该验证前，不替换
现有 GRPO workflow，也不启动正式 200-task evaluation。
