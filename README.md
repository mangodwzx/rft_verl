# RFT / RL 的大 k 值 pass@k 对比实验

本项目只研究一个问题：**经过多步训练后，RFT 与 RL 的 pass@k 差距，是否随 k 增大呈现先上升后下降的趋势？** RL 固定使用 GRPO；对照为 Base、RFT、GRPO。流程是数据准备 → 多步训练 → 同协议采样 → 曲线、差值和统计报告。

当前代码已收敛到该实验，**尚未进行 GPU 训练或真实模型评测，也没有实验结论**。检查记录见 [STATIC_REVIEW.md](STATIC_REVIEW.md)，精确定义见 [DESIGN.md](DESIGN.md)。

## 实验设置

| 项目 | 默认设置 |
|---|---|
| Qwen3 | `qwen3_1p7b`：Qwen/Qwen3-1.7B-Base；`qwen3_4b`：Qwen/Qwen3-4B-Base |
| Qwen3.5 | `qwen35_2b`：Qwen/Qwen3.5-2B-Base；`qwen35_4b`：Qwen/Qwen3.5-4B-Base |
| 方法 | Base、在线按题平均成功回答的 RFT、GRPO |
| 框架 | verl，固定提交 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` |
| 调试 | GSM8K 固定 32 题，仅检查生成和判分 |
| pilot / dev | MATH 原始 train 中分层选取 2048 题 / 独立留出 256 题 |
| full | MATH train 排除 dev 后的全部可用题；包含 pilot |
| 正式评测 | MATH-500 全部 500 题，同时单列 level 4–5 |
| 可选扩展 | DeepMath 固定 8192 或 16384 题，主实验完成后再启用 |
| 训练预算 | 3 epochs，每批 64 题，每题生成 `m=8` 条回答；每批仅更新一次 |
| 训练种子 | 42、43、44；首轮可只跑 42 |
| 优化器 | AdamW，lr=1e-6，weight decay=0，梯度裁剪 1.0 |
| 解码 | temperature=1、top-p=1、top-k=-1，仅 EOS 停止 |
| 长度 | prompt≤4096、response≤4096、context=8192 |
| 评测采样 | pilot 的 dev 和正式 MATH-500 都默认每题 `n=256` |
| 报告 k | 1、2、4、8、16、32、64、128、256 |

RFT 在每轮用当前模型生成回答，筛选正确回答计算监督损失，再继续下一轮。它不是预先生成一次数据后做离线 SFT。两种方法使用同一初始 Base、题池、采样预算和判题器；默认训练全部语言参数，Qwen3.5 视觉模块冻结。

`m` 是训练组大小，`n` 是评测时实际采样数，`k` 是估计的尝试预算。默认 `n=k=256` 时，逐题 pass@256 是这 256 条回答中是否至少成功一次，再对题目平均。要更细致地估计高 k 曲线，可增加 `n` 到 512 或 1024。

## 1. 安装

使用 Linux、CUDA GPU、Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在仓库根目录执行：

```bash
bash scripts/setup_linux.sh
source vendor/verl/.venv/bin/activate
```

脚本使用固定 verl 提交的 `uv.lock` 安装 Python 3.12 GPU 环境，再安装本项目、判题和绘图依赖；环境清单保存到 `vendor/environment.freeze.txt`。后续命令统一使用这个已激活的 Python。安装本身不启动实验。

**当前服务器使用 A800 + 570 驱动，而固定环境使用 CUDA 13。** 可以使用 NVIDIA 官方的前向兼容库，无需更换系统驱动：

```bash
# 首次配置需要；仅下载、校验并解压到项目 vendor，不安装系统驱动。
bash scripts/install_cuda_compat.sh

# 此后每个新终端/tmux 会话使用这一条，替代仅激活 .venv 的命令。
source scripts/activate.sh
```

`activate.sh` 同时设置数据盘上的 Python/包/模型缓存和临时目录，并在 535/570 驱动上为本项目的进程加载兼容库。默认运行目录为 `/mnt/public2/wangzixu/rft_runtime`；兼容库位于 `vendor/cuda-compat-13-0/`。它适用于本文的数据中心 GPU 环境，不代表任意显卡/驱动组合都受支持。[NVIDIA 兼容矩阵](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)

本机已通过真实 CUDA 分配、BF16 反向传播、FlashAttention 前后向、vLLM CuMem sleep/wake 和单 rank NCCL 检查；这不等于已验证模型生成或多卡训练。复查时，先激活上述脚本，再选择自己可用的一张 GPU，例如：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/check_gpu_runtime.py
```

兼容库必须在启动 Python 和 Ray 之前设置；已有进程不会因另一个终端的 `source` 自动更新。请从激活后的终端启动新的实验进程，不复用激活前启动的 Ray 实例。不要停止共享服务器上其他人的 Ray 任务。

训练默认使用 **8 张 GPU**。可在下面的 pipeline/train 命令中加入 `--set trainer.n_gpus_per_node=4` 调整卡数；全局回答批大小必须能被 GPU 数整除。卡数设置不保证模型和 batch 能装入显存，实际部署仍需验证。

## 2. 准备数据

```bash
python -m math_rft.data prepare --output-dir data
```

生成 `pilot/dev/full/math500/gsm8k_debug` 的 JSONL、Parquet 和 `manifest.json`。先按学科 × 难度从 MATH 原始 7500 条 train 中留出 dev，再从剩余题取 pilot。full 包含 pilot，不应再将两者拼接。去重后的实际数量以 manifest 为准。

数据来源固定到不可变 revision；保留开发集、测试集与训练数据的分离，排除与固定示例的精确重叠。已有非空输出目录不会覆盖。已有本地原始数据时，可用 `--math-train-jsonl`、`--math500-jsonl`、`--gsm8k-jsonl` 指定保留官方字段的文件。

原始 MATH train 有两条无花括号的数字答案框、两条空答案框和两条 `Level ?` 难度。数据读取支持前一种历史格式；两条空答案按题目及解答双指纹匹配 [审计修正表](configs/math_reference_corrections.json)，其数学答案均为 0。未知难度保留为 `null` 并独立分层，不猜测难度或删除题目。所有参考答案调整与未知难度题 ID 写入 `manifest.json`，原始解答保留；模型输出仍使用严格答案框判分规则。

## 3. GSM8K 流程检查

```bash
python -m math_rft.evaluate \
  --model Qwen/Qwen3-1.7B-Base \
  --data data/gsm8k_debug.jsonl --role debug --n 8 \
  --output runs/debug/gsm8k.jsonl
```

检查回答、最终 `\boxed{...}` 和判分状态。正确为 1，错误为 0；缺少 box、解析失败和超时分别记录。该步骤不训练。

## 4. 先跑一个模型、一个种子的 pilot

### 保留大 k 实验预算的低显存运行

训练、评测和流水线均支持 `--profile lowmem`。该配置保留训练 batch=64、每题 m=8、3 epochs、
prompt/response 各 4096 tokens、context=8192、原优化器与验证/保存频率；评测仍默认每题
**n=256，报告到 pass@256**。它不会缩短成两步训练，也不会减少数据、回答数或回答长度。

低显存模式只改变执行设置：生成显存预算 0.3、最大并发 16、分块 prefill 每批最多 2048 tokens、
eager 执行、参数/优化器 CPU offload，以及关闭零熵系数下的纯统计熵计算。
独立评测每次提交 8 条回答，但每题仍完整累计 256 条；同一题/样本编号的种子规则不变。
较低并发和 CPU 搬运会降低速度，不能保证不同执行设置下结果逐位一致；比较双方应统一设置。

在已安排好 GPU 的终端中运行，例如两卡 Qwen3-1.7B（卡号只是示例，启动前确认资源）：

```bash
source scripts/activate.sh
export CUDA_VISIBLE_DEVICES=6,7

# 只查看计划。
python -m math_rft.pipeline --profile lowmem --stage pilot \
  --models qwen3_1p7b --seeds 42 --n 256 \
  --set trainer.n_gpus_per_node=2

# Base 评测 → RFT 训练/评测 → GRPO 训练/评测 → 比较与绘图。
python -m math_rft.pipeline --profile lowmem --stage pilot \
  --models qwen3_1p7b --seeds 42 --n 256 \
  --set trainer.n_gpus_per_node=2 --execute
```

这仍是 2048 题、每种方法 96 次更新的 pilot，随后在全部 256 道 dev 题上评测。
结果默认存放在 `runs_lowmem_pipeline/`，与常规 `runs/` 分开。
使用 `--stage full` 时仍按 full 数据训练并评测完整 MATH-500；模型/种子默认矩阵也不因 lowmem 改变。
不显式覆盖卡数时，训练仍默认 8 卡。指定两卡只调整 FSDP 资源，保持全局 batch 和更新次数。

也可以分开训练、评测。以下命令不应与上面的完整 pipeline 重复执行：

```bash
python -m math_rft.train --profile lowmem --model qwen3_1p7b \
  --method rft --stage pilot --seed 42 \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --set trainer.n_gpus_per_node=2

python -m math_rft.train --profile lowmem --model qwen3_1p7b \
  --method grpo --stage pilot --seed 42 \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --set trainer.n_gpus_per_node=2
```

单独训练默认输出 `runs_lowmem/`。训练成功后，从完成记录读取导出路径，分别评测：

```bash
for method in rft grpo; do
  run_dir="runs_lowmem/qwen3_1p7b/$method/pilot/seed_42"
  model_path=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_hf_model"])' \
    "$run_dir/training_complete.json") || break
  python -m math_rft.evaluate --profile lowmem --model "$model_path" \
    --base-model Qwen/Qwen3-1.7B-Base --method "$method" --training-seed 42 \
    --data data/dev.jsonl --role development --n 256 \
    --output "$run_dir/dev.jsonl" || break
done
```

lowmem 自动向训练 Ray 环境及独立评测传递 `VLLM_USE_FLASHINFER_SAMPLER=0`，规避本机旧 nvcc
编译采样内核的问题。项目层熵适配会严格校验固定 verl 方法源码，保持上游 checkout 干净。
CPU offload 在训练阶段仍需把状态载回 GPU；0.3 是生成引擎预算，不是整个任务显存上限，峰值需实测。
`--set` 可覆盖训练执行参数；独立评测的 `--batch-size`、`--gpu-memory-utilization`、
`--max-num-seqs`、`--max-num-batched-tokens` 等显式参数优先于 profile 默认值。
设置或源码变更后使用新输出目录；恢复训练时保留原 `--profile` 和全部覆盖参数。

### 常规执行设置

先查看计划；没有 `--execute` 就不会下载模型或启动任务：

```bash
python -m math_rft.pipeline --stage pilot --models qwen3_1p7b --seeds 42
```

执行：

```bash
python -m math_rft.pipeline --stage pilot --models qwen3_1p7b --seeds 42 --execute
```

流水线依次评测 Base、训练 RFT/GRPO、评测终点、比较并画图。pilot 只使用独立 dev，**每题 n=256，也会报告到 pass@256**。默认 2048 题、每批 64 题、3 epochs，共 96 次更新。计划中的 `FINAL_GLOBAL_STEP` 只是展示占位符，执行时会读取成功完成标记定位最终导出的模型。

## 5. 正式跑四模型

在 dev 上确定设置后执行：

```bash
python -m math_rft.pipeline --stage full --execute
```

默认运行四模型 × 两种训练方法 × 三个种子，共 24 次训练，另有每模型的 Base 基线。评测固定为 MATH-500 全部 500 题、每题 n=256，自动给出整体和 level 4–5 曲线。

pilot 和 full **各自从同一个冻结的 Base revision 开始**，full 不接着 pilot 的参数训练。正式终点使用预先确定的训练结束检查点；不按测试集分数选检查点。模型 revision 记录在 `runs/model_revisions.json`。

要增加评测采样数，可用 `--n 512` 或 `--n 1024`，并选择新 `--output-dir`，例如：

```bash
python -m math_rft.pipeline --stage full --n 512 --output-dir runs_n512 --execute
```

该命令在新目录运行整套实验。如果只想重新评测已有训练终点，使用 `math_rft.evaluate --model <最终HF导出目录> --n 512`，为每个模型指定新的评测输出路径，并保留相同 `--base-model`、`--method`、`--training-seed` 等身份参数，随后重新 compare/plot。原输出路径不允许改变 n 后续跑。

## 查看结果

每个模型的自动报告位于：

```text
runs/comparisons/<stage>/<model>/rft_vs_grpo/
  comparison.json          各种子及汇总的配对结果
  per_seed_deltas.csv       各种子的 RFT−GRPO
  aggregate_deltas.csv      跨种子均值、两方法分数和差值区间
  trend_summary.json/csv    有符号差值和绝对差距的曲线形状、峰值位置
  high_k_changes.csv        Δ128−Δ64、Δ256−Δ128、Δ256−Δ64 及配对区间
  figures/                 整体 / level 4–5 的 pass@k 与差值图，PNG 和 PDF
```

其中 `Δk = pass@k(RFT) − pass@k(GRPO)`。图中的 RFT/GRPO 曲线是所选全部训练种子的均值；差值图同时显示单种子曲线和逐点 95% 配对 bootstrap 区间。`trend_summary` 分别描述 `Δk` 和 `|Δk|`，避免将负差值继续下降误读为差距缩小。`rise_then_fall` 只描述已报告 k 网格上的观察形状，不自动证明该规律成立。

原始输出位置：

- Base：`runs/baselines/<stage>/<model>/<dev或math500>.jsonl`。
- 训练与终点评测：`runs/<model>/<rft或grpo>/<stage>/seed_<seed>/`。
- 每个评测 JSONL 旁边都有 `.meta.json`、`.prompts.jsonl` 和 `.summary.json`，保留配置、逐题信息、实际 token IDs、失败与截断记录。

需要单独重做比较或绘图时：

```bash
python -m math_rft.analyze compare \
  --left runs/qwen3_1p7b/rft/full/seed_42/math500.jsonl \
  --right runs/qwen3_1p7b/grpo/full/seed_42/math500.jsonl \
  --output-dir runs/manual_comparison

python -m math_rft.plot \
  --comparison runs/manual_comparison/comparison.json \
  --baseline runs/baselines/full/qwen3_1p7b/math500.jsonl.summary.json \
  --output-dir runs/manual_comparison/figures
```

多种子时，`--left` 和 `--right` 各传入多个文件，并按相同种子顺序配对。

## 中断恢复与可选扩展

评测使用原命令、原路径重跑会补齐缺失样本；完整训练会被 pipeline 跳过。未完成的训练需要先显式恢复，再重启原 pipeline：

```bash
python -m math_rft.train --model qwen3_1p7b --method rft --stage pilot --seed 42 \
  --resume-from runs/qwen3_1p7b/rft/pilot/seed_42/checkpoints/global_step_40
```

恢复时复用原模型 revision 和所有 `--set` 覆盖值；源码、环境和数据也必须一致。上例只适用于实际存在第 40 步检查点的运行。最终完整 HF 导出位于 `checkpoints/global_step_N/actor/huggingface/`，无需再做模型合并。

主实验完成后再准备 DeepMath：

```bash
python -m math_rft.data prepare --output-dir data_deep --deepmath-size 8192
python -m math_rft.pipeline --stage deepmath_8192 --data-dir data_deep \
  --output-dir runs_deep --execute
```

也可改为 `16384` 和 `deepmath_16384`。固定排序使 8K 子集嵌套于 16K，排除与 MATH/dev/test/GSM8K 调试集重叠的题目。

## 本地检查

```bash
python scripts/static_check.py --ruff --report STATIC_CHECKS.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

这些检查验证代码、配置和轻量逻辑；真实数据准备、CUDA、分布式训练及模型效果需要在运行环境中验证。实验记录格式见 [records/README.md](records/README.md)。
