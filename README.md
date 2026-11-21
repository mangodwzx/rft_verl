# RFT / RL 的大 k 值 pass@k 对比实验

本项目只研究一个问题：**经过多步训练后，RFT 与 RL 的 pass@k 差距，是否随 k 增大呈现先上升后下降的趋势？** RL 固定使用 GRPO；对照为 Base、RFT、GRPO。流程是数据准备 → 多步训练 → 同协议采样 → 曲线、差值和统计报告。

已完成 Qwen3-1.7B、seed42 的旧协议 pilot 训练和 dev 测评。2026-11-03 的诊断发现 RFT 在训练后期严重短答化，并复现了原损失目标的长度偏置；修正后的协议尚需完整重训验证。历史静态检查记录见 [STATIC_REVIEW.md](STATIC_REVIEW.md)，当前精确定义见 [DESIGN.md](DESIGN.md)。

## ALFWorld 多轮训练与 pass@k

新增 `--benchmark alfworld`，使用纯文本环境进行在线 RFT / GRPO 训练和完整 episode 评测。
默认仍为数学实验。ALFWorld 默认数据目录为 `data_alfworld/`，结果目录为
`runs_alfworld_fixed/`；低显存流水线为 `runs_alfworld_lowmem_fixed_pipeline/`。

先使用现有训练 Python 安装纯文本依赖，再下载官方任务（不下载视觉模型）：

```bash
source scripts/activate.sh
uv pip install --python vendor/verl/.venv/bin/python -e '.[alfworld]'
python -m math_rft.alfworld_data download --output-dir vendor/alfworld_data
python -m math_rft.alfworld_data prepare \
  --data-root vendor/alfworld_data --output-dir data_alfworld --pilot-size 2048 --workers 8
```

依赖固定为 ALFWorld 0.4.2 / TextWorld 1.7.0。下载器只获取官方 JSON 和 TextWorld 游戏，
保存来源与校验记录；已有本地数据可直接使用 `prepare --data-root <目录>`。
数据准备会逐个重置可用任务，记录实际初始观察、文件哈希、环境版本和被过滤任务，
生成 JSONL、Parquet 和 `manifest.json`。非空目标目录不会覆盖。
`--workers` 控制独立 CPU 进程数，按可用 CPU 调整；任务排序与划分不随进程数改变。

| 输出 | 官方来源 | 用途 |
|---|---|---|
| `pilot` | `train` 按六类任务分层固定取 2048 个 | 首轮训练 |
| `full` | 全部符合官方 loader 条件的 `train` | 正式训练；包含 pilot |
| `dev` | `valid_seen` | 调参及训练期间验证 |
| `test` | `valid_unseen` | 冻结的最终测试集 |
| `debug` | 最多 32 个固定训练任务 | 环境及模型流程检查 |

实际数量以 manifest 为准；官方发布数量与本地 loader 过滤后的数量可能不同。
数据目录保存的是任务索引，原始 `vendor/alfworld_data/` 游戏文件也必须保留，
所有 Ray worker 需要能访问记录的绝对路径。任务或环境版本变化后应重新准备到新目录。

无需 GPU 即可检查环境重置和动作执行：

```bash
python -m math_rft.alfworld_data smoke \
  --data data_alfworld/debug.jsonl --actions look inventory
```

安排好 GPU 后，可以先做少量完整轨迹生成；此命令使用 debug 任务，不训练：

```bash
python -m math_rft.evaluate --benchmark alfworld --profile lowmem \
  --model Qwen/Qwen3-1.7B-Base \
  --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --data data_alfworld/debug.jsonl --role debug --n 2 \
  --output runs_alfworld_debug/base.jsonl
```

沿用同一 Base、两种方法、m=8、固定 S=4096、每批一次更新及 n=256：

```bash
# 默认仅打印计划；设置 CUDA_VISIBLE_DEVICES 并确认资源后，加 --execute 启动。
python -m math_rft.pipeline --benchmark alfworld --profile lowmem \
  --stage pilot --models qwen3_1p7b --seeds 42 \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --set trainer.n_gpus_per_node=2

# 正式训练从相同 Base 重新开始，并在完整 valid_unseen 上评测。
python -m math_rft.pipeline --benchmark alfworld --profile lowmem \
  --stage full --models qwen3_1p7b --seeds 42 43 44 \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --set trainer.n_gpus_per_node=2
```

独立训练使用 `python -m math_rft.train --benchmark alfworld ...`，独立正式评测使用
`python -m math_rft.evaluate --benchmark alfworld --data data_alfworld/test.jsonl --role formal --n 256 ...`。
`reevaluate` 从训练完成记录识别 benchmark 和交互限制，继续支持 GPU 分片与合并。

ALFWorld 的一条样本是一条从重置到结束的完整轨迹；`k` 表示独立尝试次数。
每轮模型输出一个动作，以换行或 EOS 结束，环境返回观察后继续生成；不会让模型生成环境反馈。
默认最多 50 个动作，每个动作最多 128 个 token；`--max-steps`、`--max-action-tokens`
在流水线中同时传给训练和评测。完整轨迹的 4096 token 上限包含动作和中间观察，
初始 prompt 上限 4096，总 context 上限 8192。固定 loss 尺度 S 不随实际轨迹长度改变。

当前提示协议为 `alfworld_raw_actions_v3_fewshot`。使用冻结 TextWorld 游戏的动作语法：放置物体写
`move OBJECT to LOCATION`，例如 `move potato 1 to countertop 1`；
`put OBJECT in/on LOCATION` 及其 `in`、`on` 变体不适用于这里的普通放置动作。
模型必须使用观察中的小写名称和数字 ID，每轮立即输出一行命令，不添加空行或说明。
Base、RFT、GRPO 共用同一组 3 个固定、完整的离线成功示例，涵盖放置、照明检查和加热放置；
示例的真实动作、观察、训练任务 ID 和资产哈希保存在
`src/math_rft/alfworld_demonstrations.py`。这些示例任务从新准备的任务集合中排除，
线上 rollout 不读取当前任务的专家动作或隐藏状态。

v3 改变了 prompt 和数据指纹。已有 v1/v2 数据与运行结果必须保留，重新 `prepare` 到新目录，
并为 Base、RFT、GRPO 使用新的输出目录；训练和流水线通过 `--data-dir` 指向新数据，
独立评测通过 `--data` 指向对应 JSONL。不要用旧 checkpoint 或旧 baseline ledger 续跑新协议。
真实 CPU 环境回归检查会从提示中提取放置命令，验证完整任务奖励为 1，并验证旧语法被拒绝：

```bash
RUN_ALFWORLD_INTEGRATION=1 PYTHONPATH=src python -m unittest discover \
  -s tests -p 'test_alfworld_protocol_integration.py' -v
```

此检查需要本地 `data_alfworld/debug.jsonl`、对应游戏资产和 ALFWorld 依赖；不加载模型，
普通测试发现默认跳过，显式启用后缺少依赖或资产也会跳过。

只有环境 `won` 为真才奖励 1；未完成任务的轨迹为 0。无法解析成单行动作或预算耗尽会结束轨迹，
环境/推理异常直接报错。合法格式但环境不接受的命令会返回反馈，模型可以继续尝试。
RFT 只学习成功轨迹的动作 token；观察和插入的格式 token 的 loss mask 为 0。
PDDL 原生运行时按进程复用独占后端，每次尝试重新加载并重置任务，不共享游戏状态；
这样避免上游反复创建动态库造成的资源累积。并发环境仍会占用临时存储，
请沿用 `source scripts/activate.sh` 将 `TMPDIR` 放在数据盘上。
评测保留动作、观察、终止原因、actor/observation token 数和每步种子；计算 pass@1 至 pass@256，
分析按任务类型分组。训练使用引擎的随机采样，禁止会让同组请求复用同一种子的 `full_determinism`；
这不是逐位可复现承诺。建议先检查 debug/dev 的合法动作率和成功率，小型 Base 若始终失败，
两种方法的二值奖励都可能没有有效学习信号。

## 2026-11-03：RFT 短答退化诊断与修正

旧协议的 RFT 平均生成长度从训练 step1 的约 362 tokens 降到 step70 的约 10 tokens，最终 dev 测评平均仅 8.35 tokens，99.51% 回答不超过 32 tokens。旧 lowmem 与新 throughput 测评都复现此现象。提示及其 token IDs、基座、训练数据、学习率和采样预算已核对；未发现 EOS/标签错位、损失分母或直接 train/dev 重叠问题。

确定可复现的目标问题位于 `src/math_rft/algorithms.py`：原 RFT 对每条成功回答的 NLL 除以该回答自身长度。即使长、短回答都正确，在线更新的期望梯度仍偏向短回答。新默认 **`--loss-normalization fixed --loss-scale-factor 4096`** 对完整响应求和，再除以固定常数；两种方法使用同一规则。GRPO 仍保留组内标准差归一化，是固定长度归一化变体，不是完整 Dr. GRPO。

这修正了一个已经证明的长度偏置，但不能保证真实模型不会因其他因素退化。旧 step96 权重不会因改代码而恢复，也不能把新目标的 RFT 与旧目标的 GRPO 当成同协议对照。必须从同一 Base 在新目录重训两者，并保留全部旧结果。`--loss-normalization per_response` 可在新运行中显式复现原目标；源码哈希保护仍会阻止用新代码直接续训/续评旧目录。

无需 GPU 的复现和回归检查：

```bash
python scripts/diagnose_rft_objective.py --verify-registered-loss
python -m unittest discover -s tests -v

# 从原始记录生成行为统计和PNG/PDF；output-dir 必须为新的空目录。
python -m math_rft.diagnose \
  --run-root runs_lowmem_pipeline --eval-root runs_eval_throughput_dualgpu \
  --baseline runs_lowmem_pipeline/baselines/pilot/qwen3_1p7b/dev.jsonl \
  --model qwen3_1p7b --stage pilot --seed 42 \
  --tensorboard-root tensorboard_log/math_rft \
  --output-dir runs_diagnostics/rft_collapse_check
```

本次已保存的诊断在 `runs_diagnostics/rft_collapse_20261103/`，包含 `diagnostics.json`、训练/评测 CSV、`figures/training_behavior.png` 和 `figures/evaluation_behavior.png`（另有 PDF）。精确梯度验证在 `objective_gradient_check.json`。

新协议 pilot 计划如下，固定沿用旧实验的 Base revision，避免模型仓库 HEAD 变化混入消融。默认仅打印计划；安排好两张 GPU、设置 `CUDA_VISIBLE_DEVICES` 后加 `--execute` 才启动：

```bash
python -m math_rft.pipeline --profile lowmem --stage pilot \
  --models qwen3_1p7b --seeds 42 --n 256 \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --loss-normalization fixed --loss-scale-factor 4096 \
  --output-dir runs_lowmem_fixed_pipeline \
  --set trainer.n_gpus_per_node=2
```

新运行的 manifest、completion 和测评元数据记录归一化方式与尺度，比较器拒绝混合不同/未知训练目标。训练日志也使用带 `_fixed_s4096` 的独立 TensorBoard 名称。固定尺度的默认输出为 `runs_fixed/`、低显存独立训练为 `runs_lowmem_fixed/`、低显存流水线为 `runs_lowmem_fixed_pipeline/`。

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
| 损失归一化 | 两种方法均对完整回答 token 求和，除以固定 S=4096；旧逐回答平均需显式选择 |
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

### 使用服务器本地模型，不依赖 SSH 1085 代理

Qwen3-1.7B-Base 的固定版本 `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` 已缓存在：

```text
/mnt/public2/wangzixu/rft_runtime/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
```

该缓存约 3.3 GiB，包含权重、配置和分词器，无需重新下载或复制。启动时先选择项目缓存，再开启离线模式：

```bash
source scripts/activate.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

训练和 pipeline 命令继续指定 `--model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1`；单独以 Hub ID 评测 Base 时对应 `--revision`。程序对固定版本先读取完整本地缓存，向训练/推理引擎传本地路径，同时保留原始 Hub revision 身份。离线模式下缓存缺失会明确报错，不会尝试代理；未固定的 `main` 也不会被静默替换成旧缓存。`reevaluate` 评测已导出模型时直接使用本地目录。

以后需要新模型时，在代理可用的终端中临时关闭离线模式并下载，例如：

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
hf download Qwen/Qwen3-1.7B-Base --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1
```

下载完成后重新设置离线环境变量即可。缓存 snapshot 内含指向 blobs 的符号链接，应保留缓存目录结构。Hugging Face 的[离线环境变量说明](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhuboffline)解释了缓存缺失和禁止 HTTP 请求时的行为。

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
结果默认存放在 `runs_lowmem_fixed_pipeline/`，与常规 `runs_fixed/` 和旧协议目录分开。
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

单独训练默认输出 `runs_lowmem_fixed/`。训练成功后，从完成记录读取导出路径，分别评测（以下对应默认 fixed/4096 目标）：

```bash
for method in rft grpo; do
  run_dir="runs_lowmem_fixed/qwen3_1p7b/$method/pilot/seed_42"
  model_path=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_hf_model"])' \
    "$run_dir/training_complete.json") || break
  python -m math_rft.evaluate --profile lowmem --model "$model_path" \
    --base-model Qwen/Qwen3-1.7B-Base --method "$method" --training-seed 42 \
    --training-loss-normalization fixed --training-loss-scale-factor 4096 \
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

### ALFWorld 多进程环境与保留进度迁移

ALFWorld 独立评测可加 `--env-workers 4`，让四个常驻 CPU 进程处理环境操作，
共享主评测器的一个 vLLM 引擎。环境进程使用 spawn，不加载模型；每条轨迹固定在同一进程，
保留原有 TextWorld 锁、任务状态隔离、动作规则和采样预算。默认 `0` 保留线程执行方式。
完整流水线通过 `--eval-env-workers 4` 将此设置传给 Base、RFT、GRPO 的独立评测，
不改变训练 worker 数量。评测日志同时显示每批耗时和吞吐。

已有部分评测不能直接修改配置后续写。先正常停止旧流水线，确认其进程结束、评测锁已释放，
再以新输出目录导入旧 ledger。下面命令保留每题 n=256、50 步和原模型版本：

```bash
source scripts/activate.sh
# 使用原实验已经安排的设备；启动前重新确认显存资源。
export CUDA_VISIBLE_DEVICES=0,1
python -m math_rft.pipeline --benchmark alfworld --profile lowmem \
  --stage pilot --models qwen3_1p7b --seeds 42 \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --set trainer.n_gpus_per_node=2 --eval-env-workers 4 \
  --baseline-resume-from runs_alfworld_lowmem_fixed_pipeline/baselines/pilot/qwen3_1p7b/dev.jsonl \
  --output-dir runs_alfworld_lowmem_parallel_pipeline --execute
```

迁移严格校验任务、模型、种子、样本数、环境版本、轨迹协议和推理参数，仅允许评测执行代码及
环境进程配置变化。源 ledger 和 prompt 索引按字节复制，新 metadata 的 `resume_source` 保存
完整旧 metadata、文件哈希和导入记录数；旧文件保持独立。随后只补齐缺失的任务/样本编号。
新旧调度可能产生不同的具体生成结果，metadata 明确标记旧记录前缀，不能把迁移结果声称为
全部由新执行配置生成。再次使用相同命令会校验旧记录前缀并继续未完成部分。
独立评测入口使用 `--resume-from OLD.jsonl --output NEW.jsonl --env-workers 4`。

可用 `scripts/benchmark_alfworld_pool.py` 在独立目录做线程/进程性能对照；该脚本保留
完整 episode 协议，不将短测记录加入正式实验。单纯增大 batch 无法消除进程内环境锁的排队。

流水线可为训练后的 RFT/GRPO 评测单独指定执行参数，保留 Base 的原始 lowmem 配置和迁移能力：

```bash
python -m math_rft.pipeline --benchmark alfworld --profile lowmem \
  --stage pilot --models qwen3_1p7b --seeds 42 \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --data-dir data_alfworld_tw_v3 --output-dir runs_alfworld_tw_v3_fast_pipeline \
  --set trainer.n_gpus_per_node=2 --eval-env-workers 8 \
  --trained-eval-gpus 0 6 --trained-eval-profile lowmem \
  --trained-eval-batch-size 16 --trained-eval-env-workers 8
```

该命令只打印计划，确认资源和短测吞吐后加 `--execute`。如需导入已停止的 Base 部分结果，另加
`--baseline-resume-from OLD.jsonl`；所有原有训练 `--set` 参数仍须保留。Base 仍是 batch=8、
单个推理引擎，训练仍使用原训练配置。只有训练后评测在指定物理 GPU 上各加载一个模型副本（TP=1），
按题目分片，每题仍采样 256 次；每张卡各用 8 个环境进程。两个分片都成功后才合并为原有的
`rft/grpo/pilot/seed_42/dev.jsonl`，后续比较读取合并结果。分片失败会终止本轮评测，避免合并不完整结果。

`--trained-eval-profile` 默认继承 `--profile`，`--trained-eval-env-workers` 默认继承
`--eval-env-workers`；不指定 GPU 时保持原有评测设备选择。还可设置
`--trained-eval-max-num-seqs`、`--trained-eval-max-num-batched-tokens`、
`--trained-eval-gpu-memory-utilization` 和 `--no-trained-eval-enforce-eager`。
这些选项不改变训练、样本数、种子或 episode 长度。执行配置仍写入各评测 manifest，已有评测不能
换配置后直接续写；已训练矩阵的源码请求也保持严格校验，应在首个训练开始前确定代码和配置。

### 已训练模型的高吞吐重新评测

独立评测新增 `--profile throughput`，适用于本机 Qwen3-1.7B / A800 80GB：
每次提交 256 条请求，由 vLLM 最多同时执行 64 条；已结束的请求可以在本次提交内补位。
该配置启用 CUDA Graph、分块 prefill，调度 token 预算为 8192，显存预算为 0.6，
并保留 `VLLM_USE_FLASHINFER_SAMPLER=0` 的本机兼容设置。它只改变执行参数，
仍使用 BF16、每题 n=256、回答上限 4096、上下文 8192 和原采样种子、温度、判分规则。
训练的 `--profile` 不受此设置影响。
首次启动会编译和捕获 CUDA Graph；随后每一批请求全部生成、判分并保存后才打印样本进度。

ALFWorld 重评应显式使用 `--env-workers`（例如 `--env-workers 8`），该数量对每张评测 GPU
分别生效。可先短测 `--profile lowmem --batch-size 16 --env-workers 8`；上述默认 throughput
大批次主要面向文本生成，其 ALFWorld 吞吐仍取决于环境并行度和 episode 长度。

`math_rft.reevaluate` 从已有 `training_complete.json` 读取最终模型，校验训练完成记录后，
依次重新评测 RFT 和 GRPO，再生成配对比较和 PNG/PDF 图表。它不启动训练，结果写入独立目录：

```bash
source scripts/activate.sh

# 只校验已有模型并打印计划。
python -m math_rft.reevaluate \
  --run-root runs_lowmem_pipeline --output-dir runs_eval_throughput \
  --model qwen3_1p7b --stage pilot --seed 42 --profile throughput

# 选择已空闲的 GPU；此例使用 GPU 6。RFT 和 GRPO 使用相同的新配置。
CUDA_VISIBLE_DEVICES=6 python -m math_rft.reevaluate \
  --run-root runs_lowmem_pipeline --output-dir runs_eval_throughput \
  --model qwen3_1p7b --stage pilot --seed 42 --profile throughput --execute
```

评测文件位于 `runs_eval_throughput/qwen3_1p7b/{rft,grpo}/pilot/seed_42/dev.jsonl`，
比较和图表位于 `runs_eval_throughput/comparisons/pilot/qwen3_1p7b/seed_42/rft_vs_grpo/`。
`--methods rft` 或 `--methods grpo` 可单独重评一种方法；默认两种都评测。
完整三条曲线还需按同一配置单独重评 Base；此入口默认只生成 RFT/GRPO 对比。

使用两张卡时，通过 `--gpus` 指定设备编号，每张卡加载独立的模型副本（TP=1），
按原题目顺序交错分配题目。在 256 题的 dev 集上，GPU 2 和 6 各处理 128 题，
每题仍完整生成 256 个独立样本。RFT 两个分片全部完成并合并后，再用两张卡评测 GRPO：

```bash
python -m math_rft.reevaluate \
  --run-root runs_lowmem_pipeline --output-dir runs_eval_throughput_dualgpu \
  --model qwen3_1p7b --stage pilot --seed 42 --profile throughput \
  --gpus 2 6 --n 256 --execute
```

去掉 `--execute` 可预览每张卡的命令。`--gpus` 会为每个子进程分别设置
`CUDA_VISIBLE_DEVICES`，覆盖父终端的同名变量；每卡使用相同的 throughput 配置。
分片保存到各方法目录的 `shards/shard_00000_of_00002/dev.jsonl` 和
`shards/shard_00001_of_00002/dev.jsonl`，运行日志为对应的 `.jsonl.log`。
例如查看 GRPO 的两卡进度：

```bash
tail -f runs_eval_throughput_dualgpu/qwen3_1p7b/grpo/pilot/seed_42/shards/*/dev.jsonl.log
```

合并前检查题目与样本完整覆盖、原始记录哈希、模型和执行配置一致性，再按原题目顺序
生成完整 `dev.jsonl` 及其汇总；原始分片保留，合并不会重新生成或判分。
任一分片失败或主程序收到 Ctrl+C 时，会停止本轮子进程及其推理引擎，跳过后续合并和比较。
使用相同命令可恢复各分片，完整分片跳过生成；改变分片数量时需要新输出目录。

新旧执行配置可能生成不同的具体答案，因此正式对照建议两种方法统一配置重新评测。
已有低显存评测的部分结果保留作记录，不混入新配置；不要修改旧 `.meta.json` 绕过校验。
新评测中断后，使用完全相同的命令、源码和环境重跑可补齐缺失样本，已完成的方法会直接跳过生成。
`--batch-size`、`--max-num-seqs`、`--gpu-memory-utilization`、`--max-num-batched-tokens`、
`--enforce-eager` / `--no-enforce-eager` 可覆盖高吞吐默认值，并会同时用于两种方法；
改动这些参数时应换用新输出目录。此配置不是所有模型的显存保证，其他模型需先检查峰值。

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

该命令在新目录运行整套实验。如果只想重新评测已有训练终点，使用 `math_rft.evaluate --model <最终HF导出目录> --n 512`，为每个模型指定新的评测输出路径，并保留相同 `--base-model`、`--method`、`--training-seed` 和已知的 `--training-loss-normalization` / `--training-loss-scale-factor`，随后重新 compare/plot。`reevaluate` 会自动从训练记录透传目标。原输出路径不允许改变 n 后续跑。

## 查看结果

每个模型的自动报告位于：

```text
runs_fixed/comparisons/<stage>/<model>/rft_vs_grpo/
  comparison.json          各种子及汇总的配对结果
  per_seed_deltas.csv       各种子的 RFT−GRPO
  aggregate_deltas.csv      跨种子均值、两方法分数和差值区间
  trend_summary.json/csv    有符号差值和绝对差距的曲线形状、峰值位置
  high_k_changes.csv        Δ128−Δ64、Δ256−Δ128、Δ256−Δ64 及配对区间
  figures/                 整体 / level 4–5 的 pass@k 与差值图，PNG 和 PDF
```

其中 `Δk = pass@k(RFT) − pass@k(GRPO)`。图中的 RFT/GRPO 曲线是所选全部训练种子的均值；差值图同时显示单种子曲线和逐点 95% 配对 bootstrap 区间。`trend_summary` 分别描述 `Δk` 和 `|Δk|`，避免将负差值继续下降误读为差距缩小。`rise_then_fall` 只描述已报告 k 网格上的观察形状，不自动证明该规律成立。

原始输出位置：

- Base：`runs_fixed/baselines/<stage>/<model>/<dev或math500>.jsonl`。
- 训练与终点评测：`runs_fixed/<model>/<rft或grpo>/<stage>/seed_<seed>/`（lowmem 流水线根目录为 `runs_lowmem_fixed_pipeline`）。
- 每个评测 JSONL 旁边都有 `.meta.json`、`.prompts.jsonl` 和 `.summary.json`，保留配置、逐题信息、实际 token IDs、失败与截断记录。

需要单独重做比较或绘图时：

```bash
python -m math_rft.analyze compare \
  --left runs_fixed/qwen3_1p7b/rft/full/seed_42/math500.jsonl \
  --right runs_fixed/qwen3_1p7b/grpo/full/seed_42/math500.jsonl \
  --output-dir runs_fixed/manual_comparison

python -m math_rft.plot \
  --comparison runs_fixed/manual_comparison/comparison.json \
  --baseline runs_fixed/baselines/full/qwen3_1p7b/math500.jsonl.summary.json \
  --output-dir runs_fixed/manual_comparison/figures
```

多种子时，`--left` 和 `--right` 各传入多个文件，并按相同种子顺序配对。

## 中断恢复与可选扩展

评测使用原命令、原路径重跑会补齐缺失样本；完整训练会被 pipeline 跳过。未完成的训练需要先显式恢复，再重启原 pipeline：

```bash
python -m math_rft.train --model qwen3_1p7b --method rft --stage pilot --seed 42 \
  --resume-from runs_fixed/qwen3_1p7b/rft/pilot/seed_42/checkpoints/global_step_40
```

恢复时复用原模型 revision、`--loss-normalization`、`--loss-scale-factor` 和所有 `--set` 覆盖值；源码、环境和数据也必须一致。上例只适用于实际存在第 40 步检查点的新协议运行，不能用于将旧协议续训成新协议。最终完整 HF 导出位于 `checkpoints/global_step_N/actor/huggingface/`，无需再做模型合并。

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
