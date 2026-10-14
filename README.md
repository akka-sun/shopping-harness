# Shopping Harness：ShopSimulator 长程购物 Agent 的 SFT 与可审计 Harness

这是一个面向新手的、可直接复现的 ShopSimulator 项目。本仓库只包含两部分：

- **LoRA SFT**：用经过 Reward v3 严格验收的购物轨迹训练模型；
- **Harness**：从轨迹提炼可验证的 Memory/Skill，并在运行时注入有限 guidance，帮助 Agent 改善搜索、核验和终止决策。

项目不包含 GRPO/强化学习训练。环境源码与商品索引构建脚本已随仓库提供，避免额外下载和版本漂移。

## 亮点与创新优化

1. **Action-only SFT**：只对 Assistant 的工具动作计算 loss，屏蔽用户指令和环境 observation，模型学到可执行策略而不是复述页面文本。
2. **严格终局验收**：只有完整 `gold_purchase` 且 `reward_valid=true` 的轨迹进入训练，训练集与 `data/evaluation/tasks.jsonl` 做 task_id 隔离，避免数据泄漏。
3. **证据驱动的 Harness**：Memory 只能来自单条规范化轨迹，Skill 至少交叉引用两条不同轨迹；fail-closed 校验、脱敏和原子写入保证知识库可审计。
4. **运行时最小注入**：每回合最多选择一个 Skill 和有限条 Memory，guidance 作为临时消息注入，不污染原始轨迹、不修改安全规则、工具 schema 或 Reward v3。
5. **冻结环境契约**：ShopSimulator v2.1、Observation v2、Tool schema v2 和 Reward v3 固定在同一版本，结果可复现。

## 5 分钟安装

要求 Linux、Python 3.10+、`uv`；训练 LoRA 需要 NVIDIA GPU。先执行：

```bash
git clone https://github.com/akka-sun/shopping-harness.git
cd shopping-harness
cp .env.example .env
bash scripts/setup.sh
```

`setup.sh` 会安装 SFT 依赖、创建隔离的 ShopSimulator 环境，并从仓库内置压缩包生成商品索引。无需安装 veRL、vLLM 或 Ray。

## 运行 SFT

仓库已提供训练/验证 JSONL：

```bash
bash scripts/sft.sh
```

指定自己的模型和输出目录：

```bash
python scripts/train_lora_sft.py \
  --model Qwen/Qwen3.5-2B \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output outputs/sft-demo \
  --epochs 3 --gradient-checkpointing
```

显存不足时可减少 batch size 或训练样本数；完整参数见 [`docs/sft.md`](docs/sft.md)。训练前如需采集新轨迹：

```bash
python scripts/collect_sft_data.py --help
```

## 运行 Harness 离线流程

先准备 `normalize_trajectory()` 生成的规范化轨迹 JSONL：

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

Harness 的组件说明和接入示例见 [`Harness/README.md`](Harness/README.md)。

## 目录结构

- `scripts/train_lora_sft.py`、`scripts/sft.sh`：LoRA SFT；
- `Harness/`：Memory/Skill schema、离线管线和运行时注入；
- `environments/ShopSimulator/`：冻结的环境实现与商品数据；
- `data/sft/`：训练/验证数据；`data/evaluation/`：仅用于最终评测；
- `tests/`：环境、SFT、Harness 和数据隔离测试；
- `docs/`：SFT、评测和 Harness 设计说明。

运行基础检查：

```bash
PYTHONPATH=src pytest -q tests/test_harness.py tests/test_train_lora_sft_cli.py tests/test_sft_collection.py
```

## 许可与复现边界

请遵守模型、ShopSimulator 数据和第三方依赖各自的许可证。正式 200 任务评测或长时间训练不会由安装脚本自动启动；请按需显式运行对应脚本。
