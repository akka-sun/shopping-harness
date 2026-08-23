# 单次 GRPO Run 轨迹信用与难度/失败类型感知采样设计

日期：2026-08-23  
状态：已确认，待实施计划

## 1. 目标与边界

本设计在现有 `Baseline → SFT → GRPO → Evaluation` 工作流内增加两项可独立关闭的 GRPO 能力：

1. 全训练过程启用基于 public-only 环境证据的轨迹内信用分配；
2. 前 30% optimizer step 沿用当前 task 选择，后 70% 使用静态 Probe 难度与在线可学习信号共同驱动的任务采样。

“单 epoch”在本设计中表示一次完整 GRPO run，不强制 `trainer.total_epochs=1`。总步数 `T` 取 `trainer.total_training_steps`，切换步为 `S=ceil(0.30*T)`。阶段只依据 checkpoint 可恢复的 committed optimizer `global_step` 判断。跳过 optimizer update 不增加 `global_step`。

本设计不改变 Environment v2.1 动作与终止语义、Reward v3、observation v2、tool schema v2、原始 GRPO advantage、actor loss、PPO clipping、KL、log probability、response mask、SFT 或 Evaluation。它不执行正式 GRPO、模型合并或 200-task Evaluation。

## 2. 总体架构

采用“项目模块承载算法、veRL 补丁只传递数据并调用接口”的结构。保留现有 `shopping_dynamic_sampling` 及 `select_reward_varying_groups`，其职责仍是 Reward v3 有效性、完整 uid 组和零方差组筛选。

新增两个默认关闭、相互独立的配置命名空间：

- `shopping_curriculum_sampling`：阶段、manifest、在线统计、优先级和 task 选择；
- `shopping_trajectory_credit`：证据版本、事件权重、责任归一化、token 投影和降级控制。

新增入口 `scripts/train_grpo_evidence_curriculum.py` 显式开启两项能力并设置 `switch_ratio=0.30`。原 `scripts/train_grpo.py` 的参数、命令和行为不变；不得通过隐式环境变量开启新能力。

每个 optimizer step 的数据流为：

```text
读取 committed global_step
→ 判断 baseline_30 / difficulty_70
→ 选择完整且 step 内不重复的 task_id 组
→ 每个 task 生成 rollout.n 条完整轨迹
→ Reward v3 结算与现有完整组/有效性/零方差过滤
→ 更新在线统计
→ 计算原始 GRPO trajectory advantage
→ 计算并守恒归一化回合责任倍率
→ 投影到 Assistant response token
→ 执行原 actor loss
→ optimizer update 成功后提交 global_step 与 sampler 状态
```

总 optimizer step、`rollout.n`、`max_num_gen_batches=3`、最大环境步数和 rollout 总预算与基线一致。

## 3. 模块边界

新增项目模块：

```text
src/shopping_grpo/training/grpo/
├── evidence.py
├── trajectory_credit.py
├── failure_types.py
├── difficulty/
│   ├── __init__.py
│   ├── instruction.py
│   ├── probe.py
│   └── manifest.py
└── curriculum/
    ├── __init__.py
    ├── state.py
    ├── priority.py
    └── sampler.py
```

各模块职责：

- `evidence.py`：生成、校验 `shopping-evidence-v1` public-only 证据事件；
- `trajectory_credit.py`：事件打分、token 加权守恒归一化、Assistant token 投影和降级诊断；
- `failure_types.py`：把 Reward v3 与运行时结果归一为调度失败类型；
- `difficulty/instruction.py`：GRPO 专用、只读公开指令的确定性规则评分器；不复用或修改现有 SFT 标注行为；
- `difficulty/probe.py`：复用当前 AgentLoop 和环境连接执行单次确定性 Probe；
- `difficulty/manifest.py`：数据隔离、分位点、产物 schema、版本和哈希；
- `curriculum/state.py`：在线单元统计、task 降频状态和 checkpoint 序列化；
- `curriculum/priority.py`：固定 v1 动态优先级公式；
- `curriculum/sampler.py`：阶段判断、混合概率、无放回抽样和确定性随机数。

新增入口和配置：

```text
scripts/probe_grpo_difficulty.py
scripts/train_grpo_evidence_curriculum.py
configs/grpo_evidence_curriculum.yaml
```

veRL patch 只增加五类接入点：选择候选 task、回传完整组结果、在普通 GRPO advantage 后调用 token responsibility、保存/恢复 sampler state、记录聚合诊断。核心算法不得进入 patch。

## 4. Fresh SFT Probe

### 4.1 执行协议

正式 GRPO 前，使用同一个 fresh SFT policy 对 GRPO train 1000 条任务和 validation 50 条任务各执行一次完整 rollout。Probe 只推理，不更新参数，不进入 SFT 或 GRPO loss。

解码固定为：

```yaml
temperature: 0.0
top_p: 1.0
do_sample: false
```

默认 `max_environment_steps=35`。元数据记录 SFT 模型绝对路径、模型文件集合 SHA-256、环境/Reward/observation/tool schema 版本、完整解码配置、实验 seed、最大环境步数和输入任务文件哈希。

`scripts/probe_grpo_difficulty.py --dry-run` 只验证路径、版本声明、task_id 唯一性、split 完整性、train/validation/Evaluation 零重叠和待执行命令。它不得加载模型、连接环境、执行动作、创建输出目录或正式 manifest。

### 4.2 公开指令规则难度

`task_difficulty` 由独立的 `grpo-public-instruction-v1` 评分器产生。输入只有公开用户指令，输出 `[0,1]` 分数和不包含完整指令的结构化特征摘要。

v1 特征包括：

- 约束类别数量；
- 预算/价格/尺寸/数量等数值约束；
- 品牌和型号约束；
- 规格选项轴数量；
- 兼容性要求；
- 多条件联合要求；
- 低权重、封顶的指令长度特征。

评分器使用确定性正则和词表，不调用 LLM，不读取目标 ASIN、gold 属性、隐藏匹配字段、Reward 内部特征或 Evaluation 内容。特征权重、封顶值与词表摘要均属于 scorer version；任何变化都提升版本并要求重建完整 manifest。

### 4.3 Probe 难度公式

运行时诊断直接提供：

```text
step_score  = min(action_steps / max_environment_steps, 1)
repeat_ratio = repeat_action_count / max(action_attempt_count, 1)
guard_score = min(guard_count / 3, 1)
```

固定综合公式为：

```text
probe_difficulty =
    0.45 × task_difficulty
  + 0.30 × step_score
  + 0.15 × repeat_ratio
  + 0.10 × guard_score
```

所有输入和输出均验证在 `[0,1]`，公式误差小于 `1e-8`。Reward v3 只判断 Probe 是否有效，不参与分数增减。

### 4.4 类别边界与无效隔离

只使用有效 train `probe_difficulty` 拟合约 33% 和 66% 分位点。边界移动到相邻实际分数之间，同分任务不得拆分。validation 只套用冻结边界；修改 validation 不得改变边界。

环境异常、Reward 不可验证、不完整 rollout、schema 不一致和上下文硬限制进入无效清单。Reward 可验证且 rollout 完整的 `repeat_loop` 和 `max_steps` 仍是有效 Probe。

Probe 前后都验证：

- train、validation 各自 task_id 唯一且覆盖输入全集；
- train 与 validation 零重叠；
- 两者与 `data/evaluation/tasks.jsonl` 零重叠；
- 不存在未知、缺失或重复 task_id。

任一校验失败立即终止，不自动忽略。

### 4.5 产物

正式产物分为：

- `difficulty_manifest.jsonl`：每个有效 task 的 task_id、split、四项原始/归一化输入、综合分数、类别、Probe 状态和 scorer version；
- `probe_metadata.json`：模型和输入哈希、运行时版本、解码配置、固定权重、train 分位点、manifest 哈希和隔离结果；
- `invalid_probes.jsonl`：task_id、split 和结构化无效原因。

正式数据目录不保存完整 Observation 或轨迹。可选原始轨迹只写入用户显式指定的本次 Probe 运行目录。

## 5. Public-only 证据事件

每个 Assistant generation 都分配单调、稳定的 `assistant_turn_id`。每个实际环境动作完成后产生一个 `shopping-evidence-v1` 事件；`think` 等不执行环境动作的 Assistant 回合也保留 token 区间并使用中性事件，以维持一一对应。

事件字段为：

```text
schema_version
assistant_turn_id
action_attempted
new_search_result_set
new_product_count
first_product_open
first_info_subpage
first_option_confirmation
guard_rejection
exact_repeat
no_progress
```

事件由动作前后的 runtime public snapshot 比较产生。snapshot 只包含已观察公开集合的哈希/计数、公开页面类型、已确认的公开选项集合及 Guard/重复计数。不得包含目标 ASIN、gold 属性、隐藏约束匹配、Reward v3 内部匹配特征、完整 Observation 或完整动作参数。

正向证据定义：新的搜索结果集合、新发现商品、首次打开某商品、首次查看信息子页、首次确认规格选项。负向证据定义：Guard rejection、精确重复和无进展。

`no_progress` 的 v1 定义是：动作没有产生任何正向证据，动作前后 public observation 状态指纹相同，且该动作不是终局动作。它可与 `guard_rejection` 或 `exact_repeat` 同时出现。

Observation projection 只能消费环境结果，不能反向生成证据；相同真实动作与环境状态在 projection 前后必须产生相同事件。

## 6. 轨迹内信用分配

### 6.1 回合基础分

默认事件分数：

| 事件 | 分数 |
|---|---:|
| 获得任一新证据 | +1.0 |
| 中性回合 | 0.0 |
| 无进展 | -0.5 |
| 精确重复 | -1.0 |
| Guard rejection | -1.0 |

同回合多个标记时求和后截断到 `[-1,1]`。所有分值均可配置，但配置哈希参与 checkpoint 兼容性校验。

### 6.2 方向、边界和守恒

原始 GRPO trajectory advantage 记为 `A`，回合基础分记为 `e_i`，该回合 Assistant response token 数为 `n_i`。

- `A>0` 时使用 `e_i`；
- `A<0` 时使用 `-e_i`；
- `A=0` 时不计算非零更新，所有最终 token advantage 仍为零。

先计算 token 加权中心：

```text
e_bar = Σ(n_i × signed_e_i) / Σn_i
centered_i = signed_e_i - e_bar
```

目标倍率为 `m_i = 1 + α×centered_i`，默认 `α=0.5`。若目标倍率越过默认 `[0.5,1.5]`，统一减小 `α` 到可行最大值，不逐项独立裁剪。最后使用 float64 做一次残差校正，使：

```text
Σ(n_i × m_i) / Σn_i = 1 ± 1e-6
```

校正不得越过倍率边界。若边界、有限性和守恒无法同时证明，整条轨迹倍率降级为 1。

因此责任加权不翻转 `A` 的符号，不改变每条轨迹的 response-token advantage 总质量，也不会因回合或 token 更多而提高轨迹总权重。

### 6.3 Token 投影与对齐

AgentLoop 为每个 Assistant generation 记录最终 response tensor 的半开区间 `[start,end)`。事件、`assistant_turn_id`、Assistant 回合和 token 区间必须一一对应。只有区间内且属于现有 response mask 的 Assistant 生成 token可得到：

```text
token_advantage = original_trajectory_advantage × turn_multiplier
```

Prompt、Observation、tool response、padding 和其他非 Assistant token advantage 保持 0。加权发生在普通 GRPO advantage 之后、actor loss 之前，不重新生成 rollout、Reward 或 log probability。

v1 中只要发生 context compaction，该轨迹整体降级为倍率 1，原因记为 `context_compaction_alignment_unsupported`；不删除旧 token、不重拼轨迹、不猜测边界。

事件缺失、未知版本、数量不一致、区间不完整、mask 冲突、非有限权重或 Reward 无效也整轨迹降级，不允许部分加权。

## 7. 失败类型归一化

固定映射如下：

| Reward/runtime 结果 | 调度类型 |
|---|---|
| `gold_purchase` 且 `reward_valid=true` | `strict_success` |
| `valid_alternative_purchase`, `partial_alternative_purchase` | `near_miss` |
| `wrong_purchase` | `wrong_commit` |
| `graceful_stop`, `early_abstain` | `premature_stop` |
| `repeat_loop`, `max_steps` | `exploration_stall` |
| infrastructure、Reward/schema 无效或不完整组 | `invalid` |
| Reward 有效但无法识别 | `unknown_valid` |

失败类型只影响后续采样统计，不进入 Prompt、Observation、tool schema 或 Reward，不产生训练片段。`invalid` 不更新能力统计。`unknown_valid` 更新有效轨迹/组、终局效用和方差统计，但不更新失败类型占比或 recoverability。

## 8. 在线统计

主要统计单元为 `difficulty_category × failure_type`，同时维护 difficulty 类别聚合。每个单元至少保存：有效轨迹数、有效 uid 组数、严格成功数、终局效用和、组内效用方差和、零方差组数、最近更新 step 和该失败类型计数。

task 级只保存：采样次数、最近采样 step、连续有效零方差组次数。前 30% 旁路收集统计，不改变 batch 内容、顺序、梯度或步数；后 70% 每个完整有效组继续增量更新。

成功率使用 `Beta(1,1)` 平滑。v1 不使用跨 run 历史、指数衰减或滑动窗口。无效组单独计数，不更新模型能力统计。

## 9. 后 70% 动态优先级

静态 `probe_difficulty` 与在线可学习信号分工明确：`45/30/15/10` 公式只生成 Probe 难度；后 70% 的动态优先级再结合在线统计。

对任务所属难度类别计算：

```text
variance_signal    = v / (v + 0.05)
success_uncertainty = 4p(1-p)
recoverability     = clip(near_miss_rate + 0.75×wrong_commit_rate, 0, 1)
staleness          = min(steps_since_update / 25, 1)
difficulty_signal  = task.probe_difficulty

learnability =
    0.30 × variance_signal
  + 0.25 × success_uncertainty
  + 0.25 × recoverability
  + 0.10 × staleness
  + 0.10 × difficulty_signal
```

其中 `v` 是有效组内终局效用方差均值，`p` 是平滑严格成功率。再依次施加：

- 全成功倾向因子：`1 - 0.75×p²`；
- exploration stall 且无 near-miss 的因子：下限 `0.35`；
- task 连续四次有效零方差组后的因子：`0.25`。

task 一旦出现非零方差组或 near-miss，连续零方差计数清零。最终 priority 截断到默认 `[0.05,4.0]`。hard 只通过 10% 的连续难度分量影响优先级，不能凭标签自动最高；easy 通过均匀分量和类别最低覆盖保留训练份额。

每次候选选择使用：

```text
P(task) = 0.20 × Uniform(all legal tasks)
        + 0.80 × NormalizedPriority(task)
```

均匀比例可配置但必须在 `(0,1]`。同一 optimizer step 内采用加权无放回抽样，跨 step 可重复。动态类别无候选时将质量归一到其他合法类别。任何非有限概率使当前候选批次退化为均匀抽样；连续达到阈值则终止。

随机数由实验 seed、committed `global_step` 和 generation-batch index 唯一派生。相同数据、manifest、配置、seed 和 checkpoint 必须产生相同后续 task 序列。

## 10. 阶段与现有组筛选协作

`global_step < S` 使用当前 DataLoader/task 选择算法；只旁路统计。`global_step >= S` 由 curriculum sampler 提供候选 task。恢复后直接依据 restored committed `global_step` 判断，不重新执行前 30%，整个 run 只切换一次。

每个 task 仍生成完整 `rollout.n`。现有 `sampling_invalid`、完整 uid 数量检查和零方差过滤优先执行，保留/丢弃的最小单位始终是完整 uid 组。补生成最多三个 generation batch，curriculum sampler 不能扩大预算。

新入口的前 30% 在 Probe 有效任务池上使用原算法。对相同有效任务集合和固定 seed，其 task_id 序列必须与原算法一致。关闭 curriculum 时，新入口退化为原采样；关闭 trajectory credit 时，同一固定 batch 的 advantage 和 loss 与当前实现逐元素一致。

## 11. Checkpoint 与事务语义

Sampler checkpoint schema 至少包含：

- schema version、committed `global_step` 和阶段；
- `T`、切换步、manifest 哈希和关键配置哈希；
- difficulty × failure-type 统计；
- task 采样次数、最近 step 和连续零方差次数；
- sampler 随机状态；
- 连续采样降级计数和累计诊断。

恢复前验证 manifest、配置、总步数和 schema。任何不一致都在 rollout 前终止。只有 optimizer update 成功后才提交 step 和相应 sampler 状态；checkpoint 保存训练模型状态与 sampler committed state 的同一逻辑版本，避免恢复到半步统计。

## 12. 配置默认值

`shopping_curriculum_sampling` 至少包含：

```yaml
enable: false
switch_ratio: 0.30
uniform_ratio: 0.20
manifest_path: null
priority_min: 0.05
priority_max: 4.0
zero_variance_patience: 4
save_state: true
seed: 20260823
variance_scale: 0.05
staleness_steps: 25
sampler_fallback_patience: 3
```

`shopping_trajectory_credit` 至少包含：

```yaml
enable: false
event_version: shopping-evidence-v1
positive_evidence_score: 1.0
no_progress_score: -0.5
exact_repeat_score: -1.0
guard_rejection_score: -1.0
strength: 0.5
multiplier_min: 0.5
multiplier_max: 1.5
conservation_tolerance: 1.0e-6
fallback_rate_threshold: 0.05
fallback_rate_window: 100
```

启动校验拒绝缺失/不兼容 manifest、不完整 train 覆盖、Evaluation 重叠、非法比例、非法边界、未知事件版本和 checkpoint 哈希不一致。

## 13. 日志与隐私

每个 optimizer step 记录 global step、T、阶段、S、各难度类别选中数、均匀/动态实际选择数。后 70% 额外记录类别概率与实际占比、失败类型比例、类别成功率和零方差率、priority 最小/最大/均值、降级次数及原因、task 重复率、有效组获得率和 skipped update 次数。

责任日志记录正向证据、Guard、重复和无进展比例，倍率最小/最大/均值/标准差，总质量误差，加权与降级轨迹数及原因。

日志只允许事件类型和聚合数，不写完整 Observation、完整动作参数、ASIN、gold 属性、隐藏目标信息或 Reward 内部匹配特征。

## 14. 测试与验收

测试不启动正式训练，分为：

1. Probe 单元测试：公开指令评分确定性、三个行为分数边界、`45/30/15/10` 精度、train-only 分位点、同分不拆分、三类产物 schema 与哈希；
2. 数据隔离测试：重复、缺失、未知 task_id、split/Evaluation 重叠均 fail-fast；
3. 证据测试：public snapshot 增量、正负事件、projection 不变性、隐藏字段禁止和事件版本；
4. 责任测试：正负 advantage 方向、零 advantage、倍率边界、token 加权均值、质量误差、非 Assistant mask 为零、整轨迹降级和 compaction 降级；
5. 采样测试：`S-1/S` 边界、只切换一次、混合概率非零、无放回、空类别重归一、优先级公式、零方差退火与重置、非有限值均匀降级；
6. 恢复测试：切换前后恢复、固定后续 task 序列、manifest/config/T 不一致拒绝和半步状态不提交；
7. 基线等价测试：原入口命令与 batch 选择不变；两个新能力分别关闭时行为退化；固定 batch 的 advantage/loss 逐元素一致；
8. veRL patch 契约测试：patch marker、目标版本、五类薄接入点、完整 uid 组和最多三个补生成批次。

严格成功仍只接受完整 `gold_purchase` 且 `reward_valid=true`。实施验收只运行单元、集成、dry-run 和小型合成契约测试；不自动执行 GRPO、模型合并或 200-task Evaluation。

## 15. 实施交付物

实施应交付：独立确定性 Probe 入口及 dry-run、GRPO 专用规则评分器、三类 Probe 产物、public-only 证据事件、责任归一化与 token 投影、难度/失败感知 sampler、checkpoint state、独立训练入口、隔离配置、最小 veRL patch、完整测试和更新后的 `docs/grpo.md`。

实施必须保留当前脏工作区中的用户改动，不进行无关重构、文件改名、兼容层新增或实验日志提交。
