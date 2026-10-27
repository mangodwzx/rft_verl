# 四个 Qwen Base 模型的 RFT / GRPO 数学实验

本目录包含数据准备、verl 在线训练、共同检查点梯度诊断、最高 `pass@256` 评测、配对统计和绘图代码。当前交付只做了静态检查，**没有下载模型/训练数据，没有启动训练、生成或评测，也没有实验结果**。

完整设计见 [DESIGN.md](DESIGN.md)，静态审查范围见 [STATIC_REVIEW.md](STATIC_REVIEW.md)，机器可读检查记录见 [STATIC_CHECKS.json](STATIC_CHECKS.json)。下面命令供你之后在 Linux GPU 环境执行。

项目变更见 [CHANGELOG.md](CHANGELOG.md)，后续实验记录格式见 [records/README.md](records/README.md)。

## 1. 固定设置

| 项目 | 默认值 |
|---|---|
| 模型 | Qwen3-1.7B-Base、Qwen3-4B-Base、Qwen3.5-2B-Base、Qwen3.5-4B-Base |
| 主对照 | Base、在线按题平均成功回答的 RFT、GRPO |
| 框架 | verl，固定提交 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` |
| 参数更新 | FSDP2，全语言参数；Qwen3.5 视觉模块冻结 |
| 数据 | MATH 原始 train；pilot 2048、dev 256；full 包含 pilot、排除 dev |
| 调试/扩展 | GSM8K 32 题；DeepMath 固定 8192 或 16384 题 |
| 训练种子 | 42、43、44 |
| 训练采样组大小 | `m=8`，每次更新 64 题、512 条回答 |
| 主训练预算 | 3 epochs；每批新回答仅更新一次；不重采样失败组 |
| 默认优化器 | AdamW，lr=1e-6，weight decay=0，梯度裁剪 1.0 |
| 解码 | temperature=1、top-p=1、top-k=-1、仅 EOS 停止 |
| 长度 | prompt 上限 4096、response 上限 4096、context 8192 |
| 正式评测 | MATH-500 全部 500 题，并报告 level 4–5 |
| 正式采样数 | **每题 n=1024**；可改为 n=256，但精度更粗 |
| 报告 k | 1、2、4、8、16、32、64、128、**256** |
| 主要终点 | pass@128；关键次要终点 pass@256 |

`m` 是训练时的组大小，`n` 是评测采样次数，`k` 是要估计的尝试预算；三者独立。用户要求的 `k=256` 已落实，并非把训练组大小改成 256。

模型代际与大小不是严格正交控制；主要统计量是**同一模型内部 RFT−GRPO**。Qwen3.5 同时有架构和预训练差异，不能将跨代差异单独解释为某一个因素。

## 2. 文件结构

```text
math_rft_verl/
  configs/
    models.json             四模型注册表
    experiment.json         预设实验矩阵
    verl_base.yaml          明确的训练/采样/保存配置
    verl_raw_agent.yaml     原始文本 rollout 路由
  src/math_rft/
    data.py                 来源校验、分层划分、去重、JSONL/Parquet
    prompts.py              所有模型共用的固定三示例 prompt
    reward.py               统一二值数学判分及 verl 异步适配
    algorithms.py           RFT NLL、GRPO clipped loss、优势消融
    raw_loop.py             与评测一致的原始 token 生成路径
    train.py                单次训练、配置检查、断点恢复、HF 导出
    compatibility.py        固定上游版本校验
    evaluate.py             可续跑的逐样本 vLLM 采样
    metrics.py              pass@k、配对/跨种子 bootstrap
    analyze.py              完整性验证、分组统计、CSV/JSON
    diagnose.py             独立校准和留出样本的梯度机制诊断
    pipeline.py             四模型 × 方法 × 种子编排
    plot.py                 PNG/PDF 曲线和差值置信区间
    io_utils.py             确定性 I/O 和文件指纹
  scripts/
    setup_linux.sh          安装环境；不启动实验
    static_check.py         不导入实验模块的 AST/配置/lint 检查
```

## 3. 之后安装环境

运行环境为 Linux（也可使用适配 CUDA 的 WSL2）、Python 3.12。Windows 上可以编辑和静态检查；训练栈不是原生 Windows 部署方案。

先安装 Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在本目录执行：

```bash
bash scripts/setup_linux.sh
source vendor/verl/.venv/bin/activate
```

脚本拉取固定 verl 提交，使用其原始 `uv.lock` 安装 `fsdp + vllm + math`，再安装本项目及判题/静态分析/绘图依赖。上游锁定 GPU 栈包括 torch 2.13.0、vLLM 0.29.0、Transformers 5.12.1、flash-attn 2.8.3；以该提交的 lock 为准。Math-Verify 明确固定为 0.9.0，latex2sympy2_extended 为 1.11.0。最终环境记录在 `vendor/environment.freeze.txt`。

不要用另一套 Python 启动 Ray worker。安装后直接使用已激活的解释器，不再运行会重新同步并移除本项目 overlay 的 `uv run`。若上游锁定轮子的 CUDA/驱动要求不满足，需要先部署匹配运行环境；本次未进行安装或 CUDA 验证。

## 4. 数据准备

```bash
python -m math_rft.data prepare --output-dir data
```

输出 `pilot/dev/full/diagnostic/math500/gsm8k_debug` 的 JSONL 和 Parquet，以及 `manifest.json`。

- 原始 MATH train 必须是 7500 题；按学科 × level 留出 dev=256，再从剩余数据取 pilot=2048。
- `full` 含 pilot；不能把 pilot 与 full 拼接，否则会重复训练。去重后的实际 full 数量以 manifest 为准。
- `diagnostic` 是 pilot 内固定 32 题，用于机制实验；不使用正式测试题估计权重或调参。
- MATH-500 必须完整 500 题，保留 `test/...` 原始 ID。
- 对问题做 NFKC 与空白归一化后精确去重；这不是语义去重，也不能证明预训练无污染。
- 固定 few-shot 示例是手写示例，不从任何评测划分抽取。

后续加入 DeepMath 时，准备一个新输出目录；相同 split seed 会复现原来的 MATH 划分：

```bash
python -m math_rft.data prepare --output-dir data_deep --deepmath-size 8192
# 扩展版本使用 --deepmath-size 16384；相同排序使 8K 子集嵌套在 16K 中。
```

也可以传入 `--math-train-jsonl`、`--math500-jsonl`、`--gsm8k-jsonl`、`--deepmath-jsonl`。这些文件须保留官方原始字段；本地 MATH train 仍须严格为原始 7500 条。HF 来源会解析为不可变 SHA；本地来源记录内容 SHA256。已有非空输出目录不会被覆盖。

## 5. GSM8K 流程检查

仅检查生成与判分，不进行训练：

```bash
python -m math_rft.evaluate \
  --model Qwen/Qwen3-1.7B-Base \
  --data data/gsm8k_debug.jsonl --role debug --n 8 \
  --output runs/debug/gsm8k.jsonl
```

奖励只判最终 `\boxed{...}`，正确为 1，否则为 0。缺少 box、无法解析、超时均单独记录状态；依赖错误和无效参考答案直接报错。训练判分在独立 spawn 进程中执行，避免 Math-Verify 的 SIGALRM 超时在 Ray 工作线程中失效。

## 6. 第一轮 pilot 与正式矩阵

**查看计划不会启动任何任务：**

```bash
python -m math_rft.pipeline --stage pilot --models qwen3_1p7b --seeds 42
```

计划中的 `FINAL_GLOBAL_STEP` 是展示用路径；实际执行会从成功完成标记取得精确 HF 导出路径。不要把展示文本逐条复制执行，使用下面的执行入口。

首轮验证时：

```bash
python -m math_rft.pipeline --stage pilot --models qwen3_1p7b --seeds 42 --execute
```

默认 pilot 只评估 **dev，n=64**，不会打开 MATH-500 进行调参。每个模型先保存一次 Base 基线，再训练 RFT/GRPO、评估、生成配对比较。启动时统一冻结模型 SHA 到 `runs/model_revisions.json`。

确认协议后，完整四模型、两算法、三个训练种子：

```bash
python -m math_rft.pipeline --stage full --execute
```

formal 默认 MATH-500、n=1024、最大 k=256。`full` 和 `pilot` 默认都从对应 Base 开始，是独立实验；不是继续 pilot 的参数。所有算法共享数据和初始权重。完整矩阵有 24 个训练终点，外加每模型的 Base 基线。

`--n 256` 可减少评测采样，但在 `n=k=256` 时逐题 pass@256 只剩是否至少成功一次。若改采样数、解码或源码，要使用新输出位置；续跑会拒绝不兼容记录。

## 7. 单次训练、消融与恢复

```bash
python -m math_rft.train \
  --model qwen3_1p7b --method rft --stage pilot --seed 42

# 安装 verl 后仅组合并打印配置；不下载模型，不启动 Ray。
python -m math_rft.train \
  --model qwen35_4b --method grpo_fixed --stage pilot --print-config

# 长度机制对照，单模型三个种子；仍只评估 dev。
python -m math_rft.pipeline --stage pilot --models qwen3_1p7b \
  --methods rft_fixed grpo_fixed grpo_no_std grpo_positive \
  --output-dir runs_mechanism --execute
```

可用方法及精确定义见 DESIGN。支持 `--set KEY=VALUE`，未知字段会报错；改变模型大小不会自动改变方法、损失或采样预算。调学习率时各方法使用相同候选与相同 dev 调参预算，并将候选写入不同输出目录，例如：

```bash
python -m math_rft.train --model qwen3_1p7b --method rft --stage pilot \
  --seed 42 --output-dir runs_lr3e6 --set actor_rollout_ref.actor.optim.lr=3e-6
```

训练目录为 `runs/<model>/<method>/<stage>/seed_<seed>/`，包含配置、数据/源码指纹、rollout、TensorBoard 日志及检查点。每次保存自动导出 `checkpoints/global_step_N/actor/huggingface/`，**不需要额外模型合并脚本**。只有计划的最终步骤与完整导出都通过检查，才写入 `training_complete.json`。

中断训练须显式恢复，例如 pilot 默认 96 步，在第 40 步保存后中断：

```bash
python -m math_rft.train --model qwen3_1p7b --method rft --stage pilot --seed 42 \
  --resume-from runs/qwen3_1p7b/rft/pilot/seed_42/checkpoints/global_step_40
```

恢复时需复用原来的模型 SHA 和 `--set` 值，且配置、环境、源码及数据都一致；恢复成功后可重启相同 pipeline，已完成训练不会重复执行。不要将另一实验的检查点作为断点；需要共同 SFT 起点时可用 `--local-model` 为两种方法明确指定同一个 HF 导出目录，并另记实验身份。

## 8. 共同检查点梯度诊断

这个入口在同一个检查点和同一批回答上比较原始梯度，不执行参数更新。以下使用矩阵启动时已固定的 Qwen3-1.7B revision：

```bash
REV=$(python -c 'import json; print(json.load(open("runs/model_revisions.json"))["qwen3_1p7b"]["revision"])')

python -m math_rft.evaluate --model Qwen/Qwen3-1.7B-Base --revision "$REV" \
  --data data/diagnostic.jsonl --role calibration --n 256 --seed 1701 \
  --output runs/probe/calibration.jsonl

python -m math_rft.evaluate --model Qwen/Qwen3-1.7B-Base --revision "$REV" \
  --data data/diagnostic.jsonl --role probe --n 128 --seed 1702 \
  --output runs/probe/rollouts.jsonl

python -m math_rft.diagnose --model Qwen/Qwen3-1.7B-Base --revision "$REV" \
  --calibration runs/probe/calibration.jsonl --rollouts runs/probe/rollouts.jsonl \
  --group-size 8 --output-dir runs/probe/gradients
```

默认计算所有语言参数的梯度。可显式使用 `--parameter-regex 'lm_head|layers\.27\.'` 限定子空间；报告会记录参数清单，结论也必须限定于该子空间。

校准 seed 与 probe seed 独立；probe 的前一半构建更新方向，后一半估计每题梯度投影。报告含 RFT/GRPO/独立权重匹配 GRPO 的余弦与差向量范数、正负项分解、零成功/全成功组数量、困难题投影和 pass@k 的局部斜率估计。可改 `--n` 和 `--group-size` 检查采样噪声；probe n 必须能被 `2*m` 整除，且至少 `4*m`。

这里**没有把一次梯度诊断当成实际训练胜负**。权重使用有限校准的平滑成功率，仍存在估计误差；完整训练的 AdamW、梯度裁剪和参数变化要由主实验回答。

## 9. 分析、绘图与原始输出

每次评测产生：

- `*.jsonl`：逐样本响应、分数、状态、seed、实际 response token IDs。
- `*.jsonl.prompts.jsonl`：逐题 prompt、答案与实际 prompt token IDs。
- `*.jsonl.meta.json`：模型/数据/判题器/解码/环境指纹。
- `*.jsonl.summary.json`：完整样本的逐题和整体 pass@k。

同一路径重复执行 `evaluate` 会续跑缺失样本，不会重采已有样本。程序异常通常自动清锁；强制终止留下 `.lock` 时，只在确认旧进程已结束后移除该锁。半行损坏的 JSONL 会报错，需要备份后仅移除不完整尾行。

pipeline 自动输出所有种子的比较到 `runs/comparisons/<stage>/<model>/`。也可手动：

```bash
python -m math_rft.analyze compare \
  --left runs/qwen3_1p7b/rft/full/seed_42/math500.jsonl \
  --right runs/qwen3_1p7b/grpo/full/seed_42/math500.jsonl \
  --output-dir runs/manual_comparison
```

多个种子按相同顺序传给 `--left` 与 `--right`。输出 `comparison.json`、`per_seed_deltas.csv`、`aggregate_deltas.csv`，包含整体、level 4–5、学科分组，以及有限预算下新增/丢失题目。

```bash
python -m math_rft.plot \
  --summaries runs/baselines/qwen3_1p7b/math500.jsonl.summary.json \
    runs/qwen3_1p7b/rft/full/seed_42/math500.jsonl.summary.json \
    runs/qwen3_1p7b/grpo/full/seed_42/math500.jsonl.summary.json \
  --labels Base RFT GRPO \
  --comparison runs/comparisons/full/qwen3_1p7b/rft_vs_grpo/comparison.json \
  --output-dir runs/figures/qwen3_1p7b
```

上面曲线显示 seed42，差值图若使用 pipeline 三种子比较则显示三种子汇总；图中有各自明确标注。没有训练种子间的独立重复时，不应声称置信区间涵盖训练方差。

## 10. 静态检查

```bash
python scripts/static_check.py --ruff --report STATIC_CHECKS.json
```

该命令只做 AST 编译、JSON/TOML/YAML 解析、关键配置一致性检查和 Ruff；不 import 实验模块，不运行动态测试。当前交付已做这些检查及人工跨文件/上游接口审查。实际 CUDA、分布式执行、依赖安装和模型行为仍未验证。

## 核对过的上游来源

- [固定 verl 源码](https://github.com/verl-project/verl/tree/3c6d9b43b67c74ec343d77874cd8fa6e841523da)：扩展注册、聚合损失、FSDP2、原始生成、判分调度和 HF 检查点。
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B-Base)、[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Base)、[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B-Base)、[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B-Base)。
- [Qwen3.5 模型接口](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)、[vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)。
- [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K)、[Math-Verify](https://github.com/huggingface/Math-Verify)。
