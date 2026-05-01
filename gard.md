# 操作记录

- 确认 `gard.md` 为空，开始记录后续操作核心点。

## 目标

- 使用 vLLM 框架运行 Qwen 约 9B 规模模型，并记录中间过程。
- 模型 ID 已确认：`Qwen/Qwen3.5-9B`。

## 已完成

1. 创建并激活 Python 3.12 虚拟环境：

   ```bash
   cd /root/mth/codespace/vllm
   uv venv --python 3.12 --seed --managed-python
   source .venv/bin/activate
   ```

2. 首次安装 `vllm` 时，下载 `pydantic-core==2.46.3` 失败，原因是 PyPI 连接被重置。

3. 使用清华 PyPI 镜像重新安装成功：

   ```bash
   UV_HTTP_TIMEOUT=120 uv pip install vllm --torch-backend=auto \
     --index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. 环境验证结果：

   ```text
   vllm 0.20.0
   torch 2.11.0+cu130
   cuda_available True
   cuda_device_count 2
   gpu0 NVIDIA GeForce RTX 5090
   gpu1 NVIDIA GeForce RTX 5090
   ```

5. 显存状态：

   ```text
   GPU 0: NVIDIA GeForce RTX 5090, 32607 MiB total, 32110 MiB free
   GPU 1: NVIDIA GeForce RTX 5090, 32607 MiB total, 32110 MiB free
   ```

## 注意事项

- 当前目录是 vLLM 源码目录 `/root/mth/codespace/vllm`。在这个目录里运行 Python 时，`import vllm` 会优先导入本地源码，显示 `vllm dev`，并出现 `No module named 'vllm._version'` 警告。
- 后续启动模型服务建议切到源码目录外运行，例如 `/root`，这样会导入虚拟环境中安装好的 `vllm 0.20.0`。
- 大模型文件建议放到数据盘 `/root/autodl-tmp`，不要放系统盘。

## 下一步：下载并启动 Qwen 模型

先设置环境变量：

```bash
source /root/mth/codespace/vllm/.venv/bin/activate
cd /root
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_ID=Qwen/Qwen3.5-9B
```

启动 OpenAI 兼容服务：

```bash
vllm serve "$MODEL_ID" \
  --served-model-name qwen9b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000
```

如果显存不足，优先降低上下文长度：

```bash
vllm serve "$MODEL_ID" \
  --served-model-name qwen9b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
```

服务启动后，另开一个终端验证：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen9b",
    "messages": [
      {"role": "user", "content": "用一句话介绍 vLLM。"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

## 下载 Qwen/Qwen3.5-9B

模型下载目标目录：

```bash
/root/autodl-tmp/models/Qwen3.5-9B
```

准备执行的下载命令：

```bash
source /root/mth/codespace/vllm/.venv/bin/activate
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3.5-9B \
  --local-dir /root/autodl-tmp/models/Qwen3.5-9B
```

下载结果：

```text
Download complete: 19.3GB
path: /root/autodl-tmp/models/Qwen3.5-9B
```

本地模型文件确认：

```text
model.safetensors-00001-of-00004.safetensors
model.safetensors-00002-of-00004.safetensors
model.safetensors-00003-of-00004.safetensors
model.safetensors-00004-of-00004.safetensors
config.json
tokenizer.json
chat_template.jinja
```

## 实验1：Qwen3.5-9B vLLM 推理

实验参数保持一致，便于单卡和双卡对比：

```text
model: /root/autodl-tmp/models/Qwen3.5-9B
input_len: 128
output_len: 64
batch_size: 1
num_iters: 3
num_iters_warmup: 1
max_model_len: 4096
gpu_memory_utilization: 0.85
dtype: bfloat16(auto)
```

### 1. 单卡推理

命令：

```bash
source /root/mth/codespace/vllm/.venv/bin/activate
cd /root
export HF_HOME=/root/autodl-tmp/huggingface
export CUDA_VISIBLE_DEVICES=0

vllm bench latency \
  --model /root/autodl-tmp/models/Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --input-len 128 \
  --output-len 64 \
  --batch-size 1 \
  --num-iters 3 \
  --num-iters-warmup 1 \
  --output-json /root/autodl-tmp/vllm-exp1/single_gpu_latency.json
```

关键日志：

```text
Resolved architecture: Qwen3_5ForConditionalGeneration
Using FLASH_ATTN attention backend
Loading weights took 3.63 seconds
Model loading took 17.66 GiB memory and 5.28 seconds
torch.compile took 29.12 seconds
Initial profiling/warmup run took 52.58 seconds
Available KV cache memory: 6.56 GiB
GPU KV cache size: 53,328 tokens
Maximum concurrency for 4,096 tokens per request: 36.91x
```

结果：

```text
avg_latency: 0.7397 s
p50_latency: 0.7397 s
p90_latency: 0.7404 s
p99_latency: 0.7405 s
```

### 2. 分布式推理：双卡 Tensor Parallel

命令：

```bash
source /root/mth/codespace/vllm/.venv/bin/activate
cd /root
export HF_HOME=/root/autodl-tmp/huggingface
export CUDA_VISIBLE_DEVICES=0,1

vllm bench latency \
  --model /root/autodl-tmp/models/Qwen3.5-9B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --input-len 128 \
  --output-len 64 \
  --batch-size 1 \
  --num-iters 3 \
  --num-iters-warmup 1 \
  --output-json /root/autodl-tmp/vllm-exp1/tp2_latency.json
```

关键日志：

```text
world_size=2 rank=0/1 backend=nccl
vLLM is using nccl==2.28.9
Custom allreduce is disabled because platform lacks GPU P2P capability or P2P test failed
Model loading took 8.91 GiB memory and 2.97 seconds
torch.compile took 31.33 seconds
Initial profiling/warmup run took 53.30 seconds
Available KV cache memory: 15.06 GiB
GPU KV cache size: 246,576 tokens
Maximum concurrency for 4,096 tokens per request: 169.82x
```

结果：

```text
avg_latency: 0.4326 s
p50_latency: 0.4328 s
p90_latency: 0.4331 s
p99_latency: 0.4332 s
```

### 3. 实验结果对比

```text
单卡 TP=1: avg 0.7397s, p50 0.7397s, 权重显存 17.66GiB, KV cache 53,328 tokens
双卡 TP=2: avg 0.4326s, p50 0.4328s, 单卡权重显存 8.91GiB, KV cache 246,576 tokens
```

结论：

- 当前短输入短输出场景下，双卡 TP=2 延迟从 0.7397s 降到 0.4326s，约提升 1.71 倍。
- 双卡后每张卡的权重显存占用约减半，KV cache 容量从 53,328 tokens 增加到 246,576 tokens，更适合更长上下文或更高并发。
- 分布式日志显示 custom allreduce 被禁用（形象解释见本节下方「知识点」小标题）。后续如果优化 TP 性能，应重点排查 P2P、NCCL 与 vLLM 的 custom allreduce 路径。
- 首次运行的大头时间不是推理本身，而是 torch.compile、profiling、CUDA graph capture。服务长期运行时，这部分是冷启动成本。

#### 知识点：custom allreduce 被禁用，在说什么

Tensor Parallel 时，两张卡每层都要把各自算出来的一部分结果「对一下账」，合成完整张量，这一步叫 **allreduce**（大家把数凑齐）。

- **NCCL**：可以把它想成显卡之间约好的「官方快递」——走 NVIDIA 成熟的集合通信库，一般能工作，但路线是否最短、是否走 GPU 直连，要看环境。
- **vLLM 的 custom allreduce**：可以想成「同一机箱里两卡背对背递纸条」的捷径——在 **GPU 能 P2P 直连**、且测试通过时，用自定义实现减少延迟和开销；日志里提示 **P2P 不可用或自检失败**，vLLM 就会关掉这条捷径，**退回只用常规路径（例如 NCCL）**，所以你会看到 *custom allreduce disabled*。

这不等于「双卡坏了」，而是说：**当前环境没吃到 vLLM 为 allreduce 准备的那条「近路」**；TP 仍然正确，但若 allreduce 成为瓶颈，优化方向就是 **P2P/拓扑/NCCL 配置**，以及源码里的 `custom_all_reduce.py`、`parallel_state.py`。

### 4. 后续优化重点代码

优先看这些路径：

```text
vllm/model_executor/models/qwen3_5.py
vllm/model_executor/layers/mamba/gdn_linear_attn.py
vllm/v1/attention/backends/gdn_attn.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/core/sched/scheduler.py
vllm/distributed/device_communicators/custom_all_reduce.py
vllm/distributed/parallel_state.py
```

关注方向：

- 模型结构：`qwen3_5.py`，看 `Qwen3_5ForConditionalGeneration`、decoder layer、视觉 encoder 和语言模型如何组装。
- GDN/混合架构性能：`gdn_linear_attn.py` 和 `v1/attention/backends/gdn_attn.py`，这次日志明确使用了 `Triton/FLA GDN prefill kernel`。
- 显存与 CUDA graph：`gpu_model_runner.py`，看模型加载、profiling、CUDA graph capture、编译缓存和 warmup。
- KV cache：`kv_cache_utils.py`，看 cache size、block/page 计算，以及为什么 TP=2 后容量明显增加。
- 调度：`scheduler.py`，看 chunked prefill、batching、decode 调度，对吞吐优化很关键。
- 分布式通信：`custom_all_reduce.py` 和 `parallel_state.py`，重点看为什么 custom allreduce 被禁用，以及 NCCL/P2P 能否优化。

下一组实验建议：

```text
1. 固定 batch_size=1，增加 output_len：64 -> 128 -> 256，看 decode 延迟。
2. 固定 input_len/output_len，增加 batch_size：1 -> 2 -> 4 -> 8，看吞吐和显存。
3. 对比 --max-model-len 4096/8192/32768，看 KV cache 和并发变化。
4. 启动 OpenAI API server 后使用 benchmark_serving.py 测服务吞吐，而不只看单 batch latency。
```

## 实验2：投机解码（MTP，同一 checkpoint）

Qwen3.5-9B 的 `text_config` 里带有 `mtp_num_hidden_layers`，权重里带 **多 token 预测（MTP）** 头。vLLM 里用 **`method: mtp`** 走内置投机解码，**不需要再下载小草稿模型**（与文档里 `Qwen/Qwen3-0.6B` 那种 `draft_model` 不同）。

#### MTP 投机解码原理

**1. 投机解码在干什么（和模型无关的一层逻辑）**

自回归生成本来 **每步只出一个 token**，下一步要再跑一遍大模型。投机解码的想法是：每一步先让 **更便宜的一方** 连续猜若干个候选 token，再让 **大模型（target）** 用一次或少数几次前向，去 **并行校验** 这些候选。校验通过的 token 可以 **一口气提交**，未通过的从第一个拒绝处截断、退回常规单步生成。于是：**decode 物理步数可能变少**，但多了 **草稿计算 + 校验** 的额外开销；只有「省下的算力 > 多出来的开销」时，延迟或吞吐才会变好。

**2. `draft_model` 和 `method: mtp` 的差别**

- **`draft_model`**：草稿是 **另一个 HF 模型**（通常更小），词表与结构要和 target 对齐或兼容，需要单独权重。
- **`method: mtp`（Qwen3.5 这条线）**：草稿不是外置小模型，而是 **写在同一 checkpoint 里的 MTP 子网络**（由 `mtp_num_hidden_layers` 等配置描述）。vLLM 在 `SpeculativeConfig` 里把草稿路径 **`self.model` 指回主模型目录**，因此会 **第二次加载同一 safetensors**（日志里 `Loading drafter model...`），但可 **共享 embedding / lm_head**，避免两份大词表矩阵。

**3. Qwen3.5 的 MTP 草稿在数学上在算什么（对应 `qwen3_5_mtp.py`）**

- 主干网络照常算到当前步的 **hidden states**。
- `Qwen3_5MultiTokenPredictor` 在主干 **`num_hidden_layers` 之后** 再接若干层（`mtp_num_hidden_layers`）。前向里把 **当前 token 的 embedding** 与 **主模型传来的 hidden** 各自做 RMSNorm，在特征维 **concat** 后经 **`fc`（2×hidden → hidden）** 压回隐藏维，再进入 MTP 的 decoder layer，得到用于 **多 token 预测** 的表示，最后经与主模型 **共享的 `lm_head`** 得到草稿 logits。
- 直观理解：**MTP 头学会用「词向量 + 主模型状态」去猜后续 token**；猜得好则 target 一次验证多通过几步，猜得差则大量拒绝，白算草稿。

**4. 引擎里谁先谁后（对应 `gpu_model_runner.py`）**

Worker **`load_model`** 时：**先** 加载 target，**再** `drafter.load_model(self.model)` 加载 MTP 草稿子图。运行时由 v1 调度 + `spec_decode` 路径把 **草稿 token 提议** 与 **target 验证** 串起来（接受率、拒绝长度等决定最终是否加速）。

**5. 为何你当前实验里 MTP 反而更慢（和原理一致）**

- 短 **`output_len`** 时，本来 decode 步数就少，**投机省下的步数有限**，但 **草稿 + 校验 + 额外编译/图捕获** 的固定成本仍在。
- 日志里 **KV padding** 等说明投机路径还可能改变 **KV block 布局**，使 **可用 KV token 变少**，属于要一起记录的副作用。

**6. 配置层在做什么（对应 `speculative.py`）**

- `method=="mtp"` 且未单独给草稿路径时：**草稿 checkpoint = 主模型 checkpoint**。
- 对 `qwen3_5` 文本配置会改写为 **`qwen3_5_mtp`** 形态，并把 `mtp_num_hidden_layers` 映射为内部 **`n_predict`**，以便注册到 **`Qwen3_5MTP`** 架构。

#### 代码对照（读源码时从哪几条链看）

- **CLI → 配置**：`--speculative-config` 解析为 `SpeculativeConfig`；`method=="mtp"` 且未单独指定草稿路径时，把 **草稿模型路径指回主模型 checkpoint**（同一套权重里含 MTP 头）。见 `vllm/config/speculative.py` 里 `__post_init__` 对 `mtp` 的分支。
- **Qwen3.5 + MTP 的 HF 形态**：开启投机时会把 `qwen3_5` 文本配置的 `model_type` 等改写为 **`qwen3_5_mtp`**，并把 `mtp_num_hidden_layers` 映射为内部的 `n_predict`，架构名变为 **`Qwen3_5MTP`**。见同文件 `convert_hf_text_config_for_mtp` 一段。
- **草稿网络长什么样**：`Qwen3_5MTP` 包一层 `Qwen3_5MultiTokenPredictor`：在主干 `num_hidden_layers` **之后** 接 `mtp_num_hidden_layers` 个小 transformer 层，用 `fc` 融合「当前 token 的 embedding」与「主模型最后一层 hidden」，再进 MTP layer 得到用于 **多步预测** 的表示。见 `vllm/model_executor/models/qwen3_5_mtp.py`。
- **为何日志写「共享 embedding / lm_head」**：加载完主模型后若存在 `drafter`，会 `load_model`；MTP 草稿与主模型词表与输出头一致，**直接复用主模型的 `embed_tokens` 与 `lm_head`**，避免两份大矩阵。见 `vllm/v1/spec_decode/llm_base_proposer.py` 里 `_maybe_share_embeddings` / `_maybe_share_lm_head` 的 MTP 分支。
- **为何日志出现 `Loading drafter model`**：GPU worker 在 `load_model` 里先加载 **target**，再调 `self.drafter.load_model(self.model)` 加载 **drafter（MTP 头）**。见 `vllm/v1/worker/gpu_model_runner.py`。

命令（与实验1单卡参数对齐，仅增加 `--speculative-config`）：

```bash
source /root/mth/codespace/vllm/.venv/bin/activate
cd /root
export HF_HOME=/root/autodl-tmp/huggingface
export CUDA_VISIBLE_DEVICES=0

vllm bench latency \
  --model /root/autodl-tmp/models/Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --input-len 128 \
  --output-len 64 \
  --batch-size 1 \
  --num-iters 2 \
  --num-iters-warmup 1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --output-json /root/autodl-tmp/vllm-exp1/single_gpu_mtp1_latency.json
```

日志要点：

```text
Loading drafter model...（第二次读同一套 safetensors，较快）
Detected MTP model. Sharing target model embedding weights with the draft model.
Detected MTP model. Sharing target model lm_head weights with the draft model.
Model loading took 18.12 GiB memory（无投机时约 17.66 GiB）
Add 3 padding layers, may waste at most 12.50% KV cache memory
GPU KV cache size: 41,184 tokens（无投机时约 53,328 tokens）
```

结果（`num_iters=2`，与基线同量级对比时建议把 `--num-iters` 也调成 3 再跑一轮）：

```text
avg_latency: 0.8870 s（无投机单卡约 0.7397 s）
```

结论（当前这组短序列参数下）：

- **投机解码不等于更快**：MTP 多了一步「草稿 + 校验」，若 **接受率不高** 或 **decode 步数很少**（这里 `output_len=64`），固定开销会盖过省下的算力。
- **显存与 KV**：主模型 + MTP 路径上额外编译（日志里还有 `eagle_head` 的 `torch.compile`），且出现 **padding layers** 提示，**KV cache 可放 token 数下降**，属于要纳入实验记录的副作用。
- **下一步怎么试才有收益**：把 **`output_len` 拉大**（例如 256、512）或改用 **`vllm bench serve` / 真实并发** 看 **tokens/s**；或调 `num_speculative_tokens`（例如 2，若配置允许）并配合 **指标里的接受率** 一起看。另可选 **`draft_model`**：`--speculative-config '{"model":"Qwen/Qwen3.5-4B","method":"draft_model","num_speculative_tokens":5}'`（需额外下载草稿模型，且 tokenizer/词表需兼容）。
