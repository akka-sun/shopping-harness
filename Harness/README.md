# Harness 实施说明

本目录实现 [Harness 自进化设计](../docs/plans/2026-09-14-harness-self-evolution-design.md)
和[实施文档](../docs/plans/2026-09-15-harness-self-evolution-implementation.md)
中的最小离线/运行时闭环。它不启动训练、模型服务或正式评测。

## 组件

- `contracts.py`：Memory/Skill schema、脱敏和安全校验；
- `prompts.py`：分离的单轨迹 Memory 教师请求和多轨迹 Skill 教师请求；
- `runtime.py`：trajectory-local 状态机、声明式 trigger matcher 和 guidance 渲染。
- `pipeline.py`：离线 JSONL 请求生成与 artifact 校验；不调用模型或环境。
- `integration.py`：把当前 guidance 作为单回合临时消息注入，不污染历史上下文。

Memory 必须由单条 normalized actor-visible trajectory 生成；Skill 必须引用至少
两个不同 trajectory，并可同时使用成功/失败轨迹与已验证 Memory。运行时最多选择一个
active Skill 和有限条 Memory，不把整个历史库注入 prompt。

后续接入现有 rollout 时，应在每次 `client.complete(messages, tools)` 前调用
`KnowledgeBank.match(state)`，把返回的 `RuntimeGuidance.message` 作为临时
`harness_guidance` user/message 注入；不要修改系统安全规则、工具 schema 或 Reward v3。
可直接使用 `with_guidance(messages, guidance)` 或
`complete_with_guidance(client, messages, tools, bank, state)`；注入消息不会写回
canonical trajectory history。

## 离线 dry-run

先准备由现有 `normalize_trajectory()` 产生的轨迹 JSONL。以下命令只生成教师请求，
不会发起模型请求：

```bash
PYTHONPATH=src python -m Harness memory-requests \
  --input outputs/harness/<run>/probe/normalized.jsonl \
  --output outputs/harness/<run>/memory/teacher_requests.jsonl

PYTHONPATH=src python -m Harness skill-requests \
  --trajectories outputs/harness/<run>/probe/normalized.jsonl \
  --clusters outputs/harness/<run>/skill/clusters.jsonl \
  --memories outputs/harness/<run>/memory/active.jsonl \
  --output outputs/harness/<run>/skill/teacher_requests.jsonl
```

教师模型返回 JSONL 后，先 fail-closed 校验，再决定是否进入 active bank：

```bash
PYTHONPATH=src python -m Harness validate --kind memory \
  --input outputs/harness/<run>/memory/candidates.jsonl \
  --trajectories outputs/harness/<run>/probe/normalized.jsonl \
  --accepted outputs/harness/<run>/memory/active.jsonl \
  --rejected outputs/harness/<run>/memory/rejected.jsonl
```

Skill 的 `clusters.jsonl` 每行至少包含 `cluster_id` 和两个不同的
`trajectory_ids`，例如 `{"cluster_id":"verification-1","trajectory_ids":["t1","t2"]}`。
所有输出采用原子替换；不传 `--force` 不会覆盖已有 artifact。
