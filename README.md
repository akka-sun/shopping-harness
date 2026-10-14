# Shopping Harness

面向 ShopSimulator 长程购物 Agent 的 SFT 与 Harness 工程，围绕真实环境交互、可执行动作学习、轨迹证据抽取和运行时 guidance 构建可复现链路。本仓库只保留 SFT 与 Harness，不包含 GRPO 强化学习训练。

本版本将冻结的 ShopSimulator v2.1、Reward v3、SFT 数据工程和 Harness 离线/运行时闭环纳入同一仓库，支持从数据验收到 LoRA 训练、知识候选校验和评测的持续迭代。

## 核心能力与升级

- **Action-only SFT**：只对 Assistant 工具动作计算 loss，屏蔽用户指令和环境 observation，让模型学习可执行购物策略。
- **严格轨迹验收**：仅接受完整 `gold_purchase` 且 `reward_valid=true` 的终局；训练任务与评测任务按 `task_id` 隔离，避免数据泄漏。
- **证据驱动 Harness**：Memory 只能来自单条规范化轨迹，Skill 至少引用两条不同轨迹；候选经过脱敏、schema 校验和 fail-closed 策略后才可进入 active bank。
- **最小运行时注入**：每回合只选择有限 Memory 和一个 Skill，以临时 guidance 消息帮助 Agent 决策，不污染 canonical trajectory，也不修改工具协议和 Reward v3。
- **长程购物环境**：覆盖搜索、详情核验、变体选择、约束满足、购买和合理终止，保留完整环境状态与可审计事件。
- **可复现实验**：环境、商品数据、训练/验证 JSONL、评测任务和脚本统一管理；安装脚本不会自动启动训练或正式评测。

## 技术栈

```text
环境：ShopSimulator v2.1 / Observation v2 / Tool schema v2 / Reward v3
训练：Python / PyTorch / Transformers / PEFT / LoRA / Accelerate
Harness：Memory / Skill schema / trajectory normalization / trigger matcher
评测：ShopSimulator rollout / deterministic checks / rubric-based reports
工程：uv / pytest / JSONL / SQLite product index
```

## 购物流程

```text
用户需求 → 搜索候选 → 查看详情与属性 → 核验约束与证据
                                      ↓
                             选择正确商品变体
                                      ↓
                         购买成功或证据充分后终止
                                      ↓
                规范化轨迹 → Memory / Skill → 运行时 guidance
```

## 目录结构

```text
Harness/                         # Memory/Skill schema、离线管线与运行时注入
environments/ShopSimulator/      # 冻结环境源码、商品数据与索引脚本
data/sft/                        # SFT 训练与验证 JSONL
data/evaluation/                 # 留出评测任务，不用于训练
src/shopping_grpo/               # 环境、数据、SFT 与评测 Python 包
scripts/                         # SFT、采集、评测和 Harness CLI
docs/                            # SFT、评测和 Harness 设计文档
tests/                           # 环境、SFT、Harness 和数据隔离测试
```

## 快速启动（Linux）

```bash
git clone https://github.com/akka-sun/shopping-harness.git
cd shopping-harness
cp .env.example .env
```

安装 `uv` 后执行：

```bash
bash scripts/setup.sh
```

脚本会安装 SFT 依赖、创建隔离的 ShopSimulator 环境，并从仓库内置压缩包构建商品索引。无需安装 veRL、vLLM 或 Ray。仅运行 Harness 离线校验时，可直接使用已有 Python 环境并设置 `PYTHONPATH=src`。

## LoRA SFT

仓库已提供可直接使用的训练/验证数据：

```bash
bash scripts/sft.sh
```

自定义模型和输出目录：

```bash
python scripts/train_lora_sft.py \
  --model Qwen/Qwen3.5-2B \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output outputs/sft-demo \
  --epochs 3 --gradient-checkpointing
```

训练数据采集、过滤和显存参数见 [SFT 运行指南](docs/sft.md) 与 [数据采集说明](docs/data-collection.md)。

## Harness 离线与运行时

根据规范化轨迹生成教师请求：

```bash
PYTHONPATH=src python -m Harness memory-requests \
  --input outputs/harness/run/probe/normalized.jsonl \
  --output outputs/harness/run/memory/teacher_requests.jsonl

PYTHONPATH=src python -m Harness skill-requests \
  --trajectories outputs/harness/run/probe/normalized.jsonl \
  --clusters outputs/harness/run/skill/clusters.jsonl \
  --memories outputs/harness/run/memory/active.jsonl \
  --output outputs/harness/run/skill/teacher_requests.jsonl
```

教师模型返回候选后，先执行 fail-closed 校验：

```bash
PYTHONPATH=src python -m Harness validate --kind memory \
  --input outputs/harness/run/memory/candidates.jsonl \
  --trajectories outputs/harness/run/probe/normalized.jsonl \
  --accepted outputs/harness/run/memory/active.jsonl \
  --rejected outputs/harness/run/memory/rejected.jsonl
```

运行时接入方式、触发器和消息注入边界见 [Harness 说明](Harness/README.md)。

## 评测与工程验证

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/smoke_shop_env.py
```

正式评测使用 `data/evaluation/tasks.jsonl`，并保留固定分母、终局检查和轨迹审计。请按 [评测文档](docs/evaluation.md) 显式运行；安装脚本不会自动启动长时间训练或 200 任务评测。

API 密钥、模型权重和本地实验产物不提交到仓库。ShopSimulator 商品数据与第三方模型请遵守各自许可证。
