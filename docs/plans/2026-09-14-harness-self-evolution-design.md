# Harness 自进化设计：基于 SFT Probe 的 Memory 与 Skill

## 1. 目标与边界

本方案用教师模型从 **SFT 模型的 probe 轨迹**中提取可复用知识，并由 Harness
在运行时注入；不执行 GRPO，不更新模型参数，也不再进行第二轮 SFT。

本方案的最终运行链路是：

```text
Baseline → SFT → SFT Probe → 教师 Memory → 多轨迹 Skill → Harness Evaluation
```

其中 Memory 和 Skill 必须分开生成：

- **Memory**：只看一条轨迹，提取一条通用的错误规避经验；
- **Skill**：看多条相似轨迹，提取跨任务的通用流程指导。

这里的“通用”不是保留具体商品答案，而是去除 task id、ASIN、gold option、
隐藏字段和其他任务特有内容。严格成功仍只认 ShopSimulator Environment v2.1、
Reward v3 下完整的 `gold_purchase` 且 `reward_valid=true`。

本文件是设计和实验契约，不启动训练、模型合并或 200-task evaluation。

## 2. 设计原则

### 2.1 Skill 与 Memory 的职责

```text
Skill   = 当前阶段应该遵循的通用流程
Memory  = 当前阶段容易犯的错误以及避免方式
```

Skill 例如“购买前依次核验品类、预算、品牌、型号和 option”；Memory 例如
“标题匹配不等于 option 满足，不能跳过字段核验”。Skill 决定做什么，Memory
提醒不要怎样做错。

Memory 不是轨迹摘要，也不是具体商品事实；Skill 不是单条失败轨迹的直接改写。

### 2.2 Harness 决定加载内容

模型不输出“我要调用哪个 Skill/Memory”。Harness 根据可观测的状态机和声明式
触发条件检索内容，再将短文本注入下一次模型请求。状态来自已执行工具、当前
公开 observation、可用工具、已核验字段和轨迹统计，不读取 ShopSimulator hidden
goal 或 gold 字段。

### 2.3 教师只做诊断和归纳

教师可以看 actor-visible trajectory 和经脱敏的 Reward v3 终局类别，但不能看到
gold ASIN、目标商品私有字段或未展示给 student 的环境信息。环境和确定性检查器
负责验证教师输出，不能把教师判断直接当作 reward。

## 3. 输入：SFT Probe 轨迹

### 3.1 Probe 来源

Probe 使用当前 SFT 模型、确定性解码（temperature 0、top-p 1）和现有 ShopSimulator
环境。任务只能来自 GRPO train/validation pool，必须与
`data/evaluation/tasks.jsonl` 零重叠；最终评测集不能用于 Memory/Skill 生成。

probe 的每条原始记录应保存现有 collector 产生的完整 trajectory，包括：

- `trajectory_id`、`task_id`、`messages`、`steps`；
- `blocked_tool_calls`、context/截断信息和错误信息；
- actor-visible observation 与工具参数；
- Reward v3 的脱敏终局摘要。

`scripts/probe_grpo_difficulty.py` 的 `--raw-run-dir` 现在会将每次 probe 的完整轨迹
和 `normalize_trajectory()` 结果分别 append 到 `trajectories.jsonl` 与
`normalized.jsonl`，并在 probe metadata 中记录两者 hash、SFT 模型 hash、环境/Reward/
observation/tool schema 版本。difficulty manifest 仍可生成，但不是 Memory/Skill 的
唯一输入。

### 3.2 教师可见输入

先通过现有 `normalize_trajectory()` 生成标准事件流。教师输入包括：

- 用户公开 Query；
- `actor_visible_trajectory`；
- 真实 event id、工具名、参数、公开 observation、guard rejection；
- `reward_type`、`reward_valid`、`termination_reason` 等不含目标值的终局摘要；
- 可选的确定性统计：重复动作、最大步数、非法动作和上下文错误。

教师输入不得包括：

- `purchase.asin`、gold option、hidden goal、raw audit observation；
- 任何只在评测端可见的目标字段；
- 另一条轨迹的内容（Memory 阶段严格单轨迹）。

## 4. Memory：单轨迹错误规避经验

### 4.1 生成规则

教师逐条读取 probe trajectory，每条轨迹最多生成一条 Memory，也可以输出
`null`。Memory 只在轨迹存在可归因的错误、险些犯错或明显的错误规避机会时生成；
纯粹成功且没有可提炼避错经验的轨迹输出 `null`。Memory 应定位最早可归因的错误
Turn，并将具体错误泛化成短的、可迁移的避免错误经验。

教师输出建议 schema：

```json
{
  "schema_version": "shopping-memory-v1",
  "memory_id": "mem-<stable-hash>",
  "source_trajectory_id": "...",
  "phase": "searching | candidate_review | verification | purchase | termination",
  "trigger": {
    "state": "product_opened",
    "predicates": ["requested_constraints_not_all_verified"]
  },
  "error_type": "critical_evidence_missing",
  "first_bad_event_id": "e0008",
  "lesson": "标题匹配不等于约束满足，购买前必须核验当前商品的全部相关字段。",
  "risk": "premature_or_wrong_purchase",
  "confidence": 0.86
}
```

约束：

- `lesson` 必须是通用经验，不得出现 task id、ASIN、具体商品名、具体 gold 值；
- 必须引用该轨迹中的真实 event id；
- `error_type` 必须来自现有 frozen error taxonomy；
- 不允许输出完整流程、答案或新的工具参数；
- 基础设施错误、Reward unverifiable 和教师无法归因的轨迹输出 `null`，不制造
  伪经验。

### 4.2 Memory 验证与存储

Memory 通过三道检查后才进入 active bank：

1. schema、字段长度、脱敏和 taxonomy 检查；
2. event evidence 检查：`first_bad_event_id` 必须存在且与错误类型相容；
3. 泛化检查：拒绝包含 ASIN、商品标题、数字化目标值或隐藏字段的内容。

相同或近似 Memory 可以去重，但不要求跨轨迹一致后才存储；它们仍保留各自的
`source_trajectory_id`，用于后续 Skill 归纳和回溯。

## 5. Skill：多轨迹流程归纳

### 5.1 生成规则

Skill 只能从多条相似轨迹（以及其中已验证的 Memory）中提取。输入可以同时包含
成功轨迹和失败轨迹：成功轨迹提供实际可行的流程，失败轨迹和 Memory 提供需要补
上的检查点与禁止捷径。先按 phase、状态触发条件、工具序列和失败事件模式聚类，
再由教师归纳一个通用流程。单条轨迹不能直接晋升为 Skill。

默认晋升门槛：同一模式至少 3 条独立 trajectory，且至少来自 2 个任务；若只有
一条任务的重复 rollout，不能视为跨任务证据。

教师输出建议 schema：

```json
{
  "schema_version": "shopping-skill-v1",
  "skill_id": "skill-option-verification-v1",
  "name": "option_verification_before_purchase",
  "scope": "verification",
  "trigger": {
    "state": "product_opened",
    "predicates": ["purchase_path_available"]
  },
  "procedure": [
    "读取用户明确约束",
    "逐项核验当前商品和 variant 的相关字段",
    "确认实际价格不超过预算",
    "全部满足后才允许 buy_now"
  ],
  "forbidden_shortcuts": [
    "仅凭标题匹配直接购买"
  ],
  "source_trajectory_ids": ["...", "...", "..."],
  "source_memory_ids": ["...", "...", "..."],
  "confidence": 0.91,
  "status": "candidate"
}
```

Skill 约束：

- procedure 必须是状态相关的有限步骤，不得写入任务答案；
- 只能使用当前 tool schema 和公开 observation 能执行的动作；
- 不得覆盖系统级安全规则、Action Guard 或环境终止判定；
- 一个 Skill 只解决一个流程阶段，避免生成包罗万象的长 prompt。

### 5.2 Skill 验证与版本

Skill candidate 不能直接运行。用未参与聚类的 train holdout probe 做离线回放或
重新 rollout，比较启用和禁用 Skill 的配对结果。晋升为 active 至少需要：

- 触发条件可由 Harness 确定性识别；
- 目标错误率下降；
- strict success 不下降；
- repeat loop、wrong purchase、max-step 和 guard rejection 不显著上升；
- 没有答案泄漏或非法工具调用。

Skill 使用不可变版本。新版本通过验证后替换旧版本；失败版本标记为 `rejected`
或 `retired`，不覆盖原始证据。

## 6. 运行时 Harness

### 6.1 状态机

Harness 维护 trajectory-local state，不保存 hidden goal：

```text
START
 → SEARCHING
 → CANDIDATE_FOUND
 → PRODUCT_OPENED
 → VERIFYING
 → OPTION_SELECTED
 → READY_TO_BUY
 → PURCHASED
 → TERMINATED
```

状态转换由已执行工具和最新公开 observation 驱动；guard rejection、相同 query、
无新增候选和连续无进展计数器作为风险信号。状态机不是第二个 reward，只负责
决定检索内容和注入位置。

### 6.2 检索策略

每次模型请求前：

1. 更新当前 state 和 risk predicates；
2. 选择当前 phase 下最高优先级的 active Skill，最多 1 个主 Skill；
3. 选择当前风险匹配且高置信度的 Memory，默认最多 2 条；
4. 去重、冲突检查和 token budget 检查；
5. 将内容放在本回合 observation 之后、工具调用之前的短 `harness_guidance` 消息。

Skill 优先级高于 Memory，但终止、安全和合法性规则高于二者：

```text
Action Guard / environment rule
  > terminal safety rule
  > phase Skill
  > error-avoidance Memory
```

模型不需要知道 memory_id 或 skill_id，也不能修改 SkillBank。若没有匹配项，不注入
任何内容；若 Skill 执行后仍触发相同风险，可以在下一回合 fallback 到 1 条最相似
Memory，但不连续注入历史内容。

### 6.3 触发条件

触发条件采用可审计的声明式字段，不使用自由文本分类器作为唯一依据。首版支持：

- `state` / `last_tool` / `available_tools`；
- `same_query_count`、`same_action_count`、`no_progress_steps`；
- `product_opened`、`required_fields_verified`、`variant_selected`；
- `purchase_available`、`environment_done`。

示例：

```json
{
  "state": "product_opened",
  "predicates": ["required_fields_verified == false"]
}
```

## 7. 数据与安全隔离

目录建议如下：

```text
outputs/harness/<run>/
  probe/
    trajectories.jsonl
    normalized.jsonl
    probe_metadata.json
  memory/
    candidates.jsonl
    active.jsonl
    rejected.jsonl
  skill/
    clusters.jsonl
    candidates.jsonl
    active.jsonl
    rejected.jsonl
  runtime/
    injection_diagnostics.jsonl
```

所有 artifact 记录输入 hash、教师模型/revision、prompt/schema 版本、环境版本和
生成时间。训练数据与 evaluation task id 必须隔离；任何含隐藏目标或答案的 artifact
不得进入教师输入、Memory、Skill 或运行时 prompt。

## 8. 评估设计

至少比较四组，模型、任务、解码和环境保持不变：

| 组 | 配置 |
|---|---|
| A | SFT，无 Harness 知识 |
| B | SFT + Memory |
| C | SFT + Skill |
| D | SFT + Skill + Memory |

主要指标：

- strict `gold_purchase` rate；
- `reward_valid` 和 done rate；
- wrong option / missing evidence / wrong purchase；
- repeat loop、max steps、guard rejection；
- 平均动作步数和终止效率；
- Skill/Memory 触发率、注入 token 数、误触发率；
- 按 probe difficulty、failure type 和 phase 分桶的成功率。

接受条件不是单一平均 reward：D 组必须在 strict success 不下降的前提下，减少
其目标错误类型，且不引入答案泄漏、非法动作或上下文超限。

## 9. 实施顺序

当前已落地的 standalone 实施位于仓库根目录 `Harness/`：

- `contracts.py`：Memory/Skill schema、frozen error taxonomy、事件证据、重复来源和
  goal/ASIN 脱敏校验；
- `prompts.py`：把 normalized trajectory 投影为 actor-visible teacher 输入，并严格
  分离单轨迹 Memory 与多轨迹 Skill 请求；
- `pipeline.py`：JSONL dry-run 请求生成和 accepted/rejected artifact 校验；
- `runtime.py`：trajectory-local 状态、声明式 trigger matcher、单 Skill/有限 Memory
  选择；
- `integration.py`：把 guidance 作为单回合临时消息注入，不写回历史消息。

这一步没有调用教师模型、没有启动 ShopSimulator、没有修改模型权重，也没有改写现有
GRPO/evaluation rollout。后续接入必须先通过 holdout A/B 验证，再决定是否替换默认流程。

### P0：轨迹与契约

1. 修复 probe 原始 trajectory 持久化和 metadata hash；
2. 复用 `normalize_trajectory` 和现有可见字段投影；
3. 增加 Memory/Skill schema、脱敏和触发条件验证器；
4. 增加教师请求的 dry-run fixture，不启动模型或环境。

### P1：离线知识生成

1. 对每条 probe 生成 Memory candidate；
2. 做 schema、evidence、泄漏和去重检查；
3. 聚类多条 Memory/轨迹，生成 Skill candidate；
4. 用 holdout probe 验证并版本化 active bank。

### P2：Harness 运行时

1. 实现 trajectory-local state machine；
2. 实现声明式 Skill/Memory matcher；
3. 注入 guidance 并记录 runtime diagnostics；
4. 先做 A/B/C/D 小规模验证，再申请正式评测执行。

## 10. 明确不做的事情

- 不运行 GRPO，不计算 policy advantage，不修改模型权重；
- 不通过 Memory/Skill 传递 gold 商品、ASIN 或隐藏 option；
- 不让教师直接改写 Reward v3 或环境终止结果；
- 不把单条轨迹直接升级为跨任务 Skill；
- 不把所有历史 Memory 拼进上下文；
- 不在本设计中增加旧环境、旧数据或兼容启动器。

## 11. 论文依据

- Reflexion：单轨迹失败经验的 verbal memory；
- ExpeL：从经验中抽取可检索、可迁移的策略；
- Voyager：多次经验形成可验证的 Skill Library；
- Self-Refine：教师式 critique 与修复建议的结构化循环；
- AgentOPSD：借鉴“定位最早出错 Turn”的诊断思想，但不采用其 GRPO 信用分配。
