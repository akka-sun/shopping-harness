# Harness 迭代演进：从轨迹经验到受控运行时知识

> 本文面向项目介绍，说明 Harness 每个实验版本解决了什么问题、做了什么修改，以及为什么采用这种修改方式。

## 1. 一句话介绍

Harness 是位于购物 Agent 与 ShopSimulator 之间的轻量运行时知识层：它从 SFT Probe 轨迹中提炼可复用的 Memory 与 Skill，在合适的状态下把短 guidance 临时注入模型，并用 Action Guard、购买前检查和上下文管理约束长程执行。

它不替换模型、不修改 Reward，也不读取隐藏 gold。最终成功标准始终是 ShopSimulator Environment v2.1、Reward v3 下完整的 `gold_purchase`，并且 `reward_valid=true`。

```text
SFT Probe 轨迹
    │
    ├─ 单轨迹错误归因 ──> Memory：避免某类错误
    └─ 多轨迹流程归纳 ──> Skill：执行某类流程
                              │
当前 instruction + observation + trajectory-local state
                              │
                 状态门控 + 语义门控
                              │
              临时 harness_guidance（不写回历史）
                              │
          模型动作 ──> Action Guard ──> Environment
```

## 2. 版本口径与证据边界

### 2.1 为什么不能只看 Git log

截至当前工作树，Git 主线历史主要记录了环境、SFT、GRPO 与评测基础设施；`Harness/`、Harness 脚本、设计文档和测试仍是未跟踪或未提交内容，仓库中也没有 `harness-v1` 至 `harness-v10` 的 tag。因此，下面的编号是**实验运行版本**，不是正式 Git release。

版本还原使用了四类证据：

1. `outputs/evaluation/` 中按时间保存的 v1–v10 轨迹和 summary；
2. `outputs/harness/` 中 Memory、Skill、阈值和 regenerated 产物；
3. 当前代码、测试和未提交 diff；
4. 当时的运行记录。

早期 v2–v4 没有独立 summary，主要依据连续产物和运行记录还原，不能把它们的局部诊断结果当成正式成功率。v5–v8 也同时存在 1/15 题诊断与 200 题完整评测，本文会明确区分。

### 2.2 两套编号不要混用

- **系统版本**：Harness v1–v10，表示运行时能力的演进。
- **内部产物版本**：Memory v4、Skill v5、`regenerated-v6`、threshold v1–v5，表示知识生成或阈值实验批次，不等于 Harness 系统版本。

## 3. 演进总览

| 版本 | 核心问题 | 核心变化 | 证据/结果 |
|---|---|---|---|
| v1 | 先证明知识能接入真实 rollout | 状态机、KnowledgeBank、临时 guidance、Memory/Skill 首次接入 | 完整 200 题 113/200（56.5%） |
| v2 | guidance 与模型正常决策互相干扰 | 调整注入位置和消息生命周期，保持 canonical history 干净 | 有轨迹产物，无独立 summary |
| v3 | 宽泛触发导致错误阶段注入 | 收紧 phase/trigger 匹配与注入数量 | 有轨迹产物，无独立 summary |
| v4 | guard 拒绝后仍反复使用旧页面动作 | 强化当前 observation 绑定与拒绝后的恢复提示 | 有轨迹产物，无独立 summary |
| v5 | Skill 可能比 Memory 更容易造成副作用 | 做 full 与 memory-only 消融，开始隔离 Skill 问题 | 15 题：full 8/15，memory-only 9/15 |
| v6 | 恢复提示仍可能被普通 guidance 稀释 | guard recovery 优先，恢复轮暂停普通 Harness 注入 | 诊断轨迹；未形成独立 summary |
| v7 | 搜索阶段的通用知识改变原本正确路径 | 将学习知识收缩到高风险决策点，减少搜索期干预 | 15 题 11/15 |
| v8 | 仅靠状态匹配仍会检索到语义无关知识 | 加入任务/商品/Skill 多层语义阈值和知识筛选 | 15 题 12/15；完整 200 题 121/200（60.5%） |
| v9 | 主要失败转为错误 sibling variant 和基础设施不稳定 | 重生成知识、购买 checkpoint、规格冲突校验、公开目标规格提示、任务级重试 | 统一配置从 119/200 提升到 124/200、126/200；后续 goal-options 运行约 67%，需按具体目录注明口径 |
| v10 | 长轨迹溢出与旧页面动作恢复不稳 | 成组上下文压缩、动态工具 schema、可见值 enum、当前页恢复契约 | 13 题定向验证 overflow=0；仍有策略级循环，不能当完整评测 |

## 4. v1：最小可运行 Harness

### 动机

此前系统只有固定系统提示词和模型参数，Probe 轨迹虽然暴露了重复搜索、证据缺失、预算违规和错误规格购买，却无法在不重训模型的情况下把这些经验反馈给下一次决策。第一步不是追求高分，而是验证“轨迹知识能否安全进入真实 rollout”。

### 怎么修改

- 定义 `shopping-memory-v1` 与 `shopping-skill-v1` 结构化合同。
- 用 `HarnessState` 跟踪 `searching`、`candidate_found`、`verification`、`ready_to_buy` 等公开状态。
- `KnowledgeBank` 根据 phase 和 trigger 选择最多一个 Skill 与少量 Memory。
- guidance 以临时 user message 注入单次请求，不写回 canonical history。
- 教师只能看到 actor-visible trajectory 和脱敏终局类别，不能看到隐藏 gold。

### 修改前后示例

以下是行为示意，用来说明每项修改带来的差别：

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 结构化合同 | 教师输出一段自由文本：“下次注意不要买错颜色。”系统无法可靠判断其来源和适用阶段。 | 输出带 `phase="purchase"`、`error_type="wrong_option"`、`trigger`、`source_trajectory_id` 的 `shopping-memory-v1` JSON，可先校验再加载。 |
| 显式状态机 | 系统只知道“当前仍在任务中”，无法区分搜索和购买。 | 执行 `search_products` 后进入 `searching`；打开商品后进入 `product_opened`；出现 Buy Now 且条件满足时进入 `ready_to_buy`。 |
| KnowledgeBank 选择 | 所有经验一次性放进 prompt，例如同时出现搜索、价格和终止建议。 | 当前为 `ready_to_buy` 时，只选择一个 purchase Skill 和少量 purchase Memory。 |
| 临时 guidance | “核对红色规格”被写进永久历史；模型换到另一个商品后仍看到它。 | 该提醒只加入本回合 `request_messages`；下一回合重新依据新页面决定是否注入。 |
| 教师可见范围 | 教师可能看到 `gold_asin=B012...`，生成“购买 B012...”这样的答案记忆。 | 教师只看到用户指令、公开页面、动作和脱敏结果，只能总结“购买前核对当前规格”。 |

### 为什么这样修改

确定性状态机比让模型自行决定“调用哪条知识”更可审计；临时注入则避免知识在对话历史中不断累积。结构化合同还能在上线前做泄漏、字段和来源校验。

### 结果与启示

v1 完整 200 题为 113/200（56.5%）。它证明了端到端链路可运行，但也说明“把知识接进去”不等于“知识一定有帮助”：触发范围、内容质量和消息位置都会改变模型行为。

## 5. v2：修正注入位置与消息生命周期

### 动机

v1 的 guidance 会与最新 observation、工具响应和模型原有推理顺序竞争。若把 guidance 永久写进历史，后续回合还会重复看到已经失效的状态建议。

### 怎么修改

- 将 guidance 放在最新 observation 之后、下一次工具调用之前。
- 只构造当回合的 `request_messages`，不修改 canonical `messages`。
- 保持环境结果、Action Guard 信息和工具 schema 的原始顺序。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 注入位置 | 消息顺序可能是“旧 guidance → 新 observation”，模型先看到过期建议，最后看到一大段页面内容。 | 顺序改为“最新 observation → 当回合 guidance → 模型生成”，提醒明确解释当前页面。 |
| 不污染 canonical history | `messages.append(guidance)` 后，红色商品的提醒在后续蓝色商品页面仍然存在。 | 使用 `request_messages = [*messages, guidance]`；请求结束后，原 `messages` 不包含 guidance。 |
| 保持协议顺序 | guidance 插在 assistant tool call 与 tool result 中间，可能破坏工具消息配对。 | assistant tool call 与对应 tool result 始终相邻；guidance 只加在完整工具回合之后。 |

### 为什么这样修改

知识必须解释“当前页面应该注意什么”，而不是成为永久系统规则。回合级临时消息既保留即时性，也避免 token 膨胀和过期建议污染后续页面。

### 证据说明

存在 `sft-curriculum-stage-c-harness-v2/trajectories.jsonl`，但没有独立 summary，因此只能确认该版用于消息接入迭代，不能给出可信的完整成功率。

## 6. v3：从“能触发”改为“只在正确阶段触发”

### 动机

早期知识只要 trigger 大致匹配就可能注入。搜索、候选比较、规格核验和购买是不同决策阶段；把购买建议放到搜索阶段，可能让模型过早收敛，把搜索建议放到购买阶段则会打断正确路径。

### 怎么修改

- 收紧 Memory `phase`、Skill `scope` 与 runtime state 的对应关系。
- 冻结 trigger predicate 白名单，不允许教师自由发明运行时字段。
- 限制单回合注入数量，避免多条相似建议叠加。
- 对非法 phase、未知 predicate、虚假 event 引用直接拒绝，而不是猜测修复。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| phase/scope 对齐 | 当前正在 `searching`，却注入“立即核对已选尺寸并购买”的 purchase Skill。 | `searching` 只匹配 searching/candidate-review 知识；purchase Skill 等到 `ready_to_buy` 才有资格触发。 |
| predicate 白名单 | 教师生成 `{"key":"looks_promising","op":"==","value":true}`，但运行时根本没有该字段。 | 只允许 `last_tool`、`available_tools`、`variant_selected`、`purchase_available` 等公开可计算字段。 |
| 注入数量上限 | 一次注入 8 条都在说“检查价格”的 Memory，挤占页面上下文。 | 按匹配度排序，只保留一个主 Skill 和最多两条 Memory。 |
| 保守拒绝 | Memory 写了不存在的 `phase="checkout"`，系统猜成 `purchase` 后上线。 | validator 直接写入 rejected，并记录 `phase is unsupported`，等待重生成。 |

### 为什么这样修改

这一步把自然语言经验转化为“可确定性匹配的知识”。保守拒绝比自动修正语义更安全，因为错误修正可能让一条知识在完全错误的阶段生效。

### 证据说明

存在 v3 轨迹但无独立 summary。该版的贡献主要是合同和触发语义收敛，不应把后续版本成绩倒推给 v3。

## 7. v4：让知识与当前页面绑定

### 动机

长程购物任务会翻页、进入详情子页并返回。模型可能继续使用旧页面的 ASIN、按钮或 option；普通文本提醒不足以稳定恢复，最终会连续触发 Action Guard。

### 怎么修改

- Harness 状态只由已执行动作和最新公开 observation 更新。
- 记录 rejected action，但不把被拒绝动作当成环境进展。
- 在恢复提示中强调当前可见商品、按钮和合法工具。
- 进一步整理 Memory 为 `final-v4.jsonl`，减少不安全、重复或实例化内容。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 只用真实进展更新状态 | 模型调用不存在的 `open_product("OLD-ASIN")` 被 guard 拒绝，但状态仍被误判为“已打开商品”。 | rejected action 只计入拒绝记录；状态仍保持在当前搜索页，直到合法工具真正执行。 |
| rejected action 不算进展 | 连续生成同一非法动作会重置 `no_progress_steps`，系统误以为任务在推进。 | 拒绝动作增加重复/无进展计数，不改变商品、页面或 phase。 |
| 当前页恢复提示 | 只回复“动作非法，请重试”，模型再次提交旧 ASIN。 | 回复“当前可见 ASIN 为 A1、A2；当前可用工具为 open_product、next_page”，把恢复范围绑定到新页面。 |
| Memory 清理 | Memory 写成“看到 ASIN B012 时买红色款”，只能用于一个任务且可能泄漏答案。 | 改成“购买前以当前页面 selected options 核对用户明确颜色要求”，并对重复经验去重。 |

### 为什么这样修改

错误的根源不是“模型不知道规则”，而是“动作引用的页面状态已经过期”。因此修复重点从增加通用规则转向绑定最新 observation。

### 证据说明

存在 v4 轨迹以及 Memory v4 产物，但没有可单独归因的完整 v4 summary。

## 8. v5：用消融实验识别 Skill 的副作用

### 动机

Memory 是短纠错经验，Skill 是多步流程。宽泛 Skill 更容易覆盖模型本来正确的决策，因此需要回答：问题来自整个 Harness，还是主要来自 Skill？

### 怎么修改

- 固化 Memory v4 与 Skill v5 作为同一组对照资产。
- 在同一批 15 个诊断任务上比较 full Harness 与 memory-only。
- 增加单任务 recovery 诊断，追踪错误页面动作和恢复过程。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 固定对照资产 | 两次评测分别更换 Memory、Skill 和阈值，即使分数变化也不知道原因。 | 两组都固定 Memory v4；A 组加载 Skill v5，B 组不加载 Skill，唯一变量是 Skill。 |
| full vs memory-only | 只看到 full Harness 某题失败，无法判断是 Memory 还是 Skill 造成。 | 同一题中 memory-only 成功、full 失败，就把对应 Skill 列为优先审查对象。 |
| 单任务 recovery 诊断 | 只看 summary 中 `invalid_action_limit=1`。 | 单独重跑 task 9610，逐步查看“旧按钮调用 → guard 拒绝 → 恢复提示 → 是否再次调用旧按钮”。 |

### 为什么这样修改

与其凭直觉继续改 prompt，不如用组件消融定位负贡献来源。只有知道 Skill 是否带来回归，才能决定提高阈值、隔离 Skill 或仅保留 Memory。

### 结果与启示

15 题诊断中，full 为 8/15，memory-only 为 9/15。样本太小，不能证明 Memory 普遍优于 Skill，但足以说明 Skill 并非天然增益，后续必须更严格门控。

## 9. v6：恢复指令优先于普通知识

### 动机

Action Guard 已明确指出上一个动作未执行，并给出当前合法目标；若此时再追加通用 Memory/Skill，模型可能忽略最精确的恢复信息，重复旧动作。

### 怎么修改

- guard rejection 后的下一轮暂停普通 Harness retrieval。
- 恢复消息置于请求末尾，成为最接近模型生成位置的指令。
- 只有一次合法环境动作执行成功后，才恢复正常知识检索。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 拒绝后暂停 retrieval | guard 说“ASIN OLD 不在当前页”，随后 Harness 又注入“继续检查有前景候选”，模型仍打开 OLD。 | recovery 回合不检索 Memory/Skill，只处理当前非法动作。 |
| recovery 放到末尾 | 最后一个消息是冗长商品说明，精确的 guard 原因埋在前面。 | 最后一个 user message 明确写“上个动作未执行；只可使用 A1/A2”。 |
| 合法动作后再恢复 | 模型尚未离开错误状态，普通 guidance 已重新出现。 | 模型成功执行 `open_product(A1)` 后清零 blocked count，下一回合才恢复 Harness 检索。 |

### 为什么这样修改

错误恢复需要局部、精确、短期的控制信号。通用知识在正常决策中可能有用，但不应与“上一个动作被拒绝”的事实竞争优先级。

### 证据说明

`harness-v6-diagnostic` 留下了诊断轨迹，但没有 summary；它是恢复机制过渡版，而不是正式 200 题版本。

## 10. v7：减少对搜索与候选发现的干预

### 动机

诊断发现，SFT 模型本身已经能完成大量可逆的搜索与导航；通用、高置信知识若在搜索阶段频繁出现，会让少数建议主导所有任务，改变本来正确的候选发现路径。

### 怎么修改

- 将学习知识重点收缩到高风险、接近不可逆购买的阶段。
- 同一商品只注入一次 guidance，避免反复提醒形成循环。
- 搜索和普通导航优先交给原始 SFT policy。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 收缩到高风险阶段 | 用户要“黑色耳机”；搜索页就注入“核对规格、价格、证据、终止条件”等整套流程，模型偏离原本搜索。 | 搜索阶段不注入；打开可购买商品并进入 `ready_to_buy` 后，才提示核对黑色和价格。 |
| 每商品只注入一次 | 模型每查看一次 description 再返回，都收到相同提醒，可能反复查看 description。 | 用 `harness_guided_products` 记录当前商品，第一次触发后不再重复注入。 |
| 保留原始搜索策略 | Harness 强制把查询改成固定模板，丢失用户原始关键词。 | `search_products`、翻页和打开候选仍由 SFT policy 决定，Harness 不改写其正常路径。 |

### 为什么这样修改

Harness 最有价值的地方不是替模型重新规划整条路径，而是在高代价决策前补一层经验检查。减少干预也使收益更容易归因。

### 结果与启示

15 题诊断达到 11/15，相比 v5 full 的 8/15 有改善；但这仍是小规模诊断结果。

## 11. v8：语义门控与完整评测基线

### 动机

状态相同不代表任务相同。例如所有商品都可能进入 `ready_to_buy`，但预算、规格轴和商品类别完全不同。仅靠 phase/trigger 仍会选中语义无关知识。

### 怎么修改

- 引入 `SemanticThresholdGate`，同时比较任务文本、当前商品信息和知识来源 query。
- 为 Memory 与 Skill 分别设置任务阈值、知识阈值和 resolved threshold。
- 加入 product weight，降低“任务相似但当前商品无关”的误命中。
- 增加 blocked Skill、候选 Skill 临时激活和来源轨迹要求。
- 根据对照报告选择去重后的 `selected-v8` 知识集。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 任务与来源语义比较 | 当前任务是“买鞋”，只因都处于 purchase 阶段，检索到来自“手机存储容量”的 Skill。 | 比较当前 instruction 与 Skill 来源 query；相似度低于 task threshold 时不注入。 |
| 分开设置阈值 | Memory 与长流程 Skill 共用一个低阈值，宽泛 Skill 很容易命中。 | Memory 可用较低相关性阈值，Skill 使用更高的 skill/task threshold，减少流程级误导。 |
| product weight | 用户目标都包含“黑色”，系统把黑色手机经验用于黑色鞋子。 | resolved score 同时考虑当前商品文本；商品类别不相似会拉低最终得分。 |
| blocked/candidate/source | 一条已知回归 Skill 只要状态是 active 就继续运行；candidate 也可能被误当正式知识。 | blocked ID 永不返回；candidate 仅在显式 `--activate-candidate-skills` 的 qualification run 中临时启用，并要求提供来源轨迹。 |
| selected-v8 去重 | 知识库里有三条近义的“购买前查价格”，一次可能重复注入。 | 根据 comparison 和内容相似度保留代表项，其余进入 rejected/未选择集合。 |

### 为什么这样修改

状态门控回答“现在能否使用这一类知识”，语义门控回答“这条知识是否真的与当前任务相关”。两层缺一不可。

### 结果与启示

- 15 题诊断：12/15。
- 完整 200 题：121/200（60.5%）。相对无 Harness 对照 120/200，仅净增 1 题：2 个 gain、1 个 regression。
- `selected-v8`：120/200（60.0%）。筛选后的更小知识集没有带来净提升。
- threshold v3–v5 与 memory-only 完整运行约为 114–116/200，说明单纯调阈值或删 Skill 不能稳定解决问题。

v8 的关键结论是：语义检索减少了无关注入，但主要瓶颈已从“检索错知识”转向“购买前选择了错误 sibling variant”。

## 12. v9：从“检索知识”转向“保护不可逆购买”

### 动机

v8 后的失败集中在 `partial_alternative_purchase` 与 `wrong_purchase`：模型经常找到正确或相近商品，却在颜色、尺寸、包装、型号等规格上买错。与此同时，评测还受 ShopSimulator 502、租约竞态和 vLLM tool parser 配置影响，需要先把基础设施噪声与策略失败分开。

### 怎么修改

#### 12.1 重生成和清理知识

- 从 1,050 条非 evaluation Probe 轨迹中确定性抽样 200 条。
- 重新生成、规范化、校验和去重 Memory/Skill，形成 `regenerated-v6` active bank。
- 保留来源轨迹、prompt/schema 版本与状态，候选 Skill 仅在资格运行中临时激活。

#### 12.2 购买前 checkpoint

- 每个商品首次进入 `ready_to_buy` 时追加一次 `[PURCHASE_CHECKPOINT]`。
- 要求比较当前 selected options、当前价格与用户显式要求。
- checkpoint 独立于语义检索：即使没有 Memory 命中，也不能跳过不可逆动作前检查。

#### 12.3 规格冲突校验与恢复

- `purchase_option_conflict()` 只使用 instruction 与当前 observation，检查显式规格是否和已选 option 冲突。
- 明确冲突时拒绝 `buy_now`，返回当前页面可见的精确匹配值。
- 正确 option 选中后，下一轮只暴露 `buy_now`，防止模型再次切回错误 sibling。
- 后续 checkpoint 加入 reset 返回的公开 `goal_options`；这不是 hidden gold，而是环境公开给 actor 的目标规格。

#### 12.4 评测可靠性

- 修正 vLLM tool-call parser，避免 XML tool call 被当作普通文本。
- 增加任务切片、指定 task、基础设施有限重试和失败审计行。
- 基础设施失败不再被当作已完成任务；重试时重新创建环境并释放旧租约。

### 修改前后示例

#### 知识重生成

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 确定性抽样 | 每次随手选择不同轨迹生成知识，两个版本无法公平比较。 | 固定 `size=200`、`seed=20260918`，每次都从同一批非 evaluation Probe 轨迹生成。 |
| 校验和去重 | 教师输出缺少 schema、引用不存在事件或多条 Memory 内容重复，仍可能进入运行库。 | candidates 依次经过规范化、schema/来源/泄漏校验和去重，分别进入 active、rejected、null。 |
| 来源与状态 | 只保存 lesson，无法知道来自哪条轨迹，也无法区分 candidate 和 active。 | artifact 保留 source IDs、prompt/schema 版本、confidence 与 status；candidate 只在资格评测临时激活。 |

#### 购买前 checkpoint

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 首次 ready-to-buy 检查 | 页面默认选中“白色 M”，用户要求“黑色 L”；模型看到 Buy Now 后直接购买。 | 首次进入 `ready_to_buy` 时提示比较当前 selected options、价格和所有显式要求，要求先改成“黑色 L”。 |
| 每商品一次 | 每个回合都重复 checkpoint，模型可能一直检查而不购买。 | `purchase_checkpointed_products` 保证同一商品只提示一次。 |
| 独立于检索 | 语义相似度略低，没有 Memory 命中，于是完全没有购买核验。 | 即使 guidance 为空，状态机到达 `ready_to_buy` 仍强制添加 checkpoint。 |

#### 规格冲突与恢复

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 冲突检测 | instruction 要“卡通-永结同心”，当前选中“酒红纯色梳”，`buy_now` 仍被执行。 | guard 从当前 visible options 找到精确要求，返回 `buy_option_conflict` 并拒绝购买。 |
| 精确匹配恢复 | 只告诉模型“规格不对”，模型可能随机选择另一个错误款。 | 返回 `visible_exact_match=卡通-永结同心`，下一步要求选择该公开值。 |
| 修正后锁定购买 | 正确 option 选中后，模型又查看其他选项并切回默认款。 | 修正完成的下一轮只暴露 `buy_now`；若模型输出其他可逆动作，也折叠为安全购买动作。 |
| 公开 goal options | checkpoint 只能从自然语言猜测“礼盒装”对应页面上的哪个长标签。 | 把 reset 已公开的目标标签加入提示：“如果该标签当前可见，选择完全相同的标签”；仍不读取 hidden gold。 |

#### 评测可靠性

| 修改 | 修改前 | 修改后 |
|---|---|---|
| tool-call parser | 模型输出 `<tool_call>...`，`hermes` parser 将其当普通文本，任务以 `assistant_final` 结束。 | 使用匹配 Qwen XML 格式的 parser，`search_products` 被解析成真实工具调用。 |
| 切片与指定任务 | 为诊断一个失败题也要从头跑 200 题。 | 可用 `--task-start/--task-end` 做四分片，或重复传 `--task-id 8187` 做单题复现。 |
| 有限基础设施重试 | 首次 reset 返回 HTTP 502，程序立即停止，且失败行可能让 resume 跳过该题。 | 仅对基础设施错误有限重试；每次重建环境，失败行保留审计但不算 completed task。 |

### 为什么这样修改

购买是不可逆动作，最适合放确定性检查；搜索是可逆且多样的，过强干预容易回归。把规则集中在最后一刻，可以在较少改变模型探索行为的前提下阻止最常见的错误购买。

### 结果与边界

- 初版 regenerated-v6、仅 ready-to-buy 检索：119/200（59.5%）。
- 加入统一 purchase checkpoint：124/200（62.0%）。
- 公开规格冲突校验的两次完整统一运行：126/200（63.0%）与 125/200（62.5%）。
- `option-hybrid` 为 127/200，但它只替换了旧运行中的失败任务，配置不统一，不能作为正式验收。
- goal-options 的后续四分片运行记录约为 134/200（67%）；对外引用时必须同时给出对应目录和合并口径，不能与 hybrid 结果混写。

v9 还验证了若干无效方向：把生成上限从 256 提到 512、在 option-selected 阶段增加宽泛提示、通用 loop recovery，都没有稳定收益或出现回归，因此没有作为主方案保留。

## 13. v10：上下文压缩与 schema 级恢复

### 动机

v9 增加 checkpoint、recovery 和 guidance 后，长轨迹更容易接近上下文上限；同时，文本形式的恢复提示仍无法阻止模型生成旧 ASIN 或旧 option。问题需要从“再写一句提示”升级到上下文结构和工具 schema 两层治理。

### 怎么修改

- 使用真实 tokenizer 计算上下文，而不是字符数估算。
- 压缩时只删除完整的 assistant tool-call + tool-result 组，避免破坏消息配对。
- 保留用户约束、最新 observation、当前 option/价格、最近 guidance 和恢复消息。
- `_guard_recovery_tools()` 动态复制工具 schema：
  - `open_product.asin` enum 只保留当前页可见 ASIN；
  - `select_option.value` enum 只保留当前页可见 option；
  - 当前不可用工具不再暴露。
- `_guard_recovery_message()` 明确列出当前合法签名、可见 ASIN 与按钮。

### 修改前后示例

| 修改 | 修改前 | 修改后 |
|---|---|---|
| 真实 tokenizer 计数 | 用字符数猜 token；中文、JSON 和工具 schema 的比例不同，可能估少后被 vLLM 拒绝。 | 通过实际 tokenizer 计算输入 token，在超过安全预算前触发压缩。 |
| 成组压缩 | 单独删除旧 tool result，却保留其 assistant tool call，消息协议失配。 | 把“assistant tool call + 对应 tool result”作为一个原子组共同删除。 |
| 保留关键证据 | 从最早消息开始粗暴截断，连用户预算或最新 selected options 一起丢失。 | 固定保留用户目标、最新页面、当前规格/价格、最近 guidance 和 recovery，只移除较早工具回合。 |
| 动态工具集合 | 在商品详情页仍向模型暴露 `search_products`、旧页 `open_product` 等全部工具。 | 根据当前 observation 只暴露此页合法工具；不可用工具从 recovery schema 中移除。 |
| ASIN enum | `open_product.asin` 仍是任意字符串，模型可以再次生成旧页 `OLD-ASIN`。 | schema 改为 `enum: ["A1", "A2"]`，只允许当前搜索页可见商品。 |
| option enum | `select_option.value` 接受任意文本，如不存在的“深黑色”。 | schema 改为当前页按钮值，例如 `enum: ["黑色", "白色"]`。 |
| 精确 recovery message | 只提示“请使用合法工具”，模型不知道哪些值合法。 | 消息列出 `open_product: asin (required)`、当前可见 A1/A2 和按钮黑色/白色。 |

### 为什么这样修改

上下文压缩解决“模型看不到或服务装不下”，动态 schema 解决“模型仍能生成当前页面不存在的值”。前者保留最新证据，后者直接缩小动作空间，比自然语言提醒更确定。

### 结果与边界

13 个历史问题任务的定向验证中，`context_overflow_tasks=0`，其中 3 个长轨迹实际触发压缩；说明明确的上下文溢出已被解决。但仍有 4 个任务因旧页面动作或合法动作内循环触发 `invalid_action_limit`。

`v10-context-guard-rerun` 是四分片诊断/重跑产物，不应在未统一合并核验前作为新的完整 200 题正式成绩。v10 解决了基础设施与动作空间问题，尚未解决商品身份判断和策略级循环。

## 14. 每轮迭代背后的统一方法

Harness 的演进并不是不断增加 prompt，而是逐步缩小控制范围：

1. **v1–v2：先接通**——知识能进入 rollout，且不污染永久历史。
2. **v3–v4：再约束**——只在正确阶段、针对当前页面生效。
3. **v5–v6：做归因**——用消融识别 Skill 副作用，让精确恢复优先。
4. **v7–v8：少干预**——退出搜索阶段，用语义门控降低误命中。
5. **v9：守住不可逆动作**——把主要能力集中到购买前规格核验。
6. **v10：结构化治理**——用成组压缩和动态 schema 处理长上下文与旧状态动作。

这条路线体现了一个核心判断：**模型已有的搜索能力应尽量保留，Harness 应重点干预高风险、可验证、不可逆的决策。**

## 15. 当前系统的组成

| 模块 | 作用 | 关键文件 |
|---|---|---|
| Probe | 采集并规范化 actor-visible 轨迹，检查 evaluation 零重叠 | `scripts/probe_grpo_difficulty.py` |
| Contract | 校验 schema、枚举、来源、泄漏与 trigger | `Harness/contracts.py` |
| Teacher prompt | 分别生成单轨迹 Memory 与多轨迹 Skill | `Harness/prompts.py` |
| Offline pipeline | 请求构造、候选校验、accepted/rejected 分流 | `Harness/pipeline.py` |
| Runtime state | 维护 trajectory-local 状态并匹配知识 | `Harness/runtime.py` |
| Semantic gate | 按任务、商品、来源 query 与阈值过滤 | `Harness/semantic.py` |
| Injection | 生成单回合临时 guidance | `Harness/integration.py` |
| Rollout safety | purchase checkpoint、Action Guard、恢复与压缩 | `src/shopping_grpo/evaluation/rollout.py` |

## 16. 对外介绍时应如何表述结果

推荐说法：

> Harness 从“全流程知识注入”逐步演进为“状态与语义双重门控、购买前重点干预”的运行时系统。v8 完整 200 题为 60.5%；v9 的购买 checkpoint 和公开规格校验把统一配置提升到 62%–63%，后续 goal-options 配置运行达到约 67%。v10 进一步消除了定向长轨迹中的上下文溢出，但商品身份判断与策略级循环仍是主要剩余问题。

不应这样表述：

- 不把 15 题或 13 题诊断成功率称作 Final-200 成绩；
- 不把 hybrid 拼接结果称作统一配置评测；
- 不把 `candidate` Skill 称为已经 holdout 晋升的 active Skill；
- 不把公开 `goal_options` 描述成隐藏 gold；
- 不声称每个 v1–v10 都对应一个 Git commit 或 tag。

## 17. 可追溯证据

- 设计：`docs/plans/2026-09-14-harness-self-evolution-design.md`
- 实施：`docs/plans/2026-09-15-harness-self-evolution-implementation.md`
- Harness 说明：`Harness/README.md`
- v1：`outputs/evaluation/sft-curriculum-stage-c-harness/summary-initial-v1.json`
- v5 消融：`outputs/evaluation/harness-v5-diagnostic/`
- v7/v8 诊断：`outputs/evaluation/harness-v7-diagnostic/`、`harness-v8-diagnostic/`
- v8 完整评测：`outputs/evaluation/sft-curriculum-stage-c-harness-v8/`
- v8 对照：`outputs/evaluation/sft-curriculum-stage-c-harness-v8/comparison.json`
- v9 checkpoint：`outputs/evaluation/sft-curriculum-stage-c-harness-v9-checkpoint-merged/`
- v9 option verifier：`outputs/evaluation/v9-final-optionverify/`、`v9-final-optionverify-r2/`
- v10 定向验证：`outputs/evaluation/v9-context-guard-validation/`
- v10 分片重跑：`outputs/evaluation/v10-context-guard-rerun/`

## 18. 当前限制与下一步

1. 为 Harness v1–v10 建立正式 Git tag 或 changelog，避免继续依赖输出目录还原版本。
2. 将 candidate、validated、active、rejected、retired 生命周期在文件命名和运行时状态上完全统一。
3. 用未参与知识生成的 holdout 做逐条知识晋升，而不是只做整库 A/B。
4. 为商品身份、多规格完整性与价格证据建立更细的公开证据检查器。
5. 对 v10 做一次单一配置、唯一任务集合、统一 protocol 的 Final-200 汇总，再确定正式对外成绩。
