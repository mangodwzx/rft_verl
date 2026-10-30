# CUDA 13 前向兼容检查

本次仅检查运行环境，不包含模型训练或 pass@k 实验结果。

- 系统：Ubuntu 22.04 x86_64，NVIDIA A800，内核驱动保持 570.172.08。
- Python：3.12.16；PyTorch：2.13.0+cu130；vLLM：0.29.0；FlashAttention：2.8.3。
- 官方兼容包：`cuda-compat-13-0_580.196.07-1ubuntu1_amd64.deb`。
- 来源：`https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/`。
- SHA256：`b0ee4fb1f439e919ca5c5fd7c5cff3e8b8afdbdb0b5ee38f7c09c047ea41a035`，与官方 Packages.gz 元数据一致。
- 安装方式：`dpkg-deb -x` 解压到项目 `vendor/cuda-compat-13-0`；没有执行系统安装、ldconfig、驱动更新或重启。

复查命令（GPU 编号按自己的资源分配选择）：

```bash
source scripts/activate.sh
CUDA_VISIBLE_DEVICES=1 python -u scripts/check_gpu_runtime.py
```

实际检查已全部通过：

1. CUDA 初始化、张量分配和读取；进程实际加载项目目录中的 `libcuda.so.580.196.07`。
2. BF16 矩阵乘法与反向传播，输出和梯度有限。
3. FlashAttention BF16 前向与反向传播。
4. vLLM CuMem 分配、sleep/wake 和数据恢复；实际分配池大小 2 MiB。
5. 单 rank NCCL all_reduce。

脚本强制使用实际 CUDA 初始化，避免 vLLM 导入后的 NVML-only 检查把“可枚举 GPU”误判为“可执行 CUDA”。测试进程已正常退出。没有启动 Ray、下载模型或运行训练；模型生成、Ray 多进程、跨 GPU 通信仍未验证。

此前 `uv pip check` 的两项元数据冲突本次未修改：上游主动覆盖的 cuSPARSELt 0.8.0 与 torch 声明的 0.8.1，以及 NumPy 2.4.6 与 mistral-common 的 `<2.4` 约束。上述微量 GPU 检查通过并不证明全部模型/依赖路径兼容。
