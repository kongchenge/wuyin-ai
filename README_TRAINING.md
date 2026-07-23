# Wuyin QLoRA 微调训练指南

基于 Qwen2.5-1.5B 的 QLoRA (Quantized Low-Rank Adaptation) 微调完整指南。

---

## 目录

- [硬件要求](#硬件要求)
- [数据格式](#数据格式)
- [本地训练指南](#本地训练指南)
- [Google Colab 训练指南](#google-colab-训练指南)
- [合并 Adapter 与基础模型](#合并-adapter-与基础模型)
- [推理测试](#推理测试)
- [常见错误排查](#常见错误排查)
- [文件说明](#文件说明)

---

## 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU 显存 | 6 GB VRAM | 8+ GB VRAM |
| 系统内存 | 16 GB RAM | 32 GB RAM |
| 磁盘空间 | 10 GB 可用 | 20 GB SSD |
| GPU 型号 | GTX 1060 6GB / T4 | RTX 3060+ / A10 / A100 |

**说明**: 1.5B 模型经过 4-bit 量化后，基础模型仅占用约 1 GB 显存，加上 LoRA adapter（r=8）和训练开销，总计约 4-6 GB。

---

## 数据格式

训练数据文件 `wuyin_train.jsonl`，每行一个 JSON 对象，格式如下：

```jsonl
{"messages": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "我是吴音，一个经过微调的 AI 助手。"}]}
{"messages": [{"role": "user", "content": "今天天气怎么样？"}, {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。建议你查看天气预报应用。"}]}
{"messages": [{"role": "system", "content": "你是一个有用的助手。"}, {"role": "user", "content": "帮我写一首诗"}, {"role": "assistant", "content": "春风拂面来，桃花朵朵开..."}]}
```

**注意事项**:
- 每条数据必须包含 `messages` 字段，值为对话列表
- `role` 支持 `system`、`user`、`assistant`
- 建议至少 100-500 条高质量数据
- 确保 JSON 格式有效，每行一个完整对象（不是 JSON 数组）

---

## 本地训练指南

### 1. 环境准备

**创建虚拟环境（推荐）**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install torch transformers datasets accelerate peft bitsandbytes trl ninja packaging

# 可选: 安装 flash-attention 加速训练（需要 CUDA 工具链）
pip install flash-attn --no-build-isolation
```

或者一键安装:
```bash
pip install -r requirements.txt
```

### 3. 准备数据

将你的训练数据保存为 `wuyin_train.jsonl`，放在项目根目录（与 `train_wuyin.py` 同级）。

### 4. 开始训练

```bash
python train_wuyin.py
```

训练过程:
1. 加载 Qwen2.5-1.5B 基础模型（自动下载，首次约 3-5 分钟）
2. 应用 4-bit 量化 + LoRA adapter
3. 加载并格式化训练数据
4. 开始 3 个 epoch 训练（T4 约 20-60 分钟）
5. 保存 adapter 到 `wuyin-lora-adapter/` 目录

### 5. 训练参数调整

在 `train_wuyin.py` 文件开头的配置区域可调整:

```python
MODEL_ID = "Qwen/Qwen2.5-1.5B"           # 或 "Qwen/Qwen2.5-1.5B-Instruct"
NUM_EPOCHS = 3                             # 训练轮数，数据少可增加到 5-10
BATCH_SIZE = 2                             # 显存不足时改为 1
GRAD_ACCUM_STEPS = 4                       # 有效批次 = BATCH_SIZE × GRAD_ACCUM_STEPS
MAX_SEQ_LENGTH = 2048                      # 最大序列长度，显存不足时减小
LEARNING_RATE = 2e-4                       # 学习率
LORA_R = 8                                 # LoRA 秩，增大可提升容量但增加显存
LORA_ALPHA = 16                            # LoRA 缩放因子，通常为 r 的 2 倍
```

---

## Google Colab 训练指南

### 1. 上传数据到 Google Drive

1. 打开 [Google Drive](https://drive.google.com)
2. 将 `wuyin_train.jsonl` 上传到 `我的云端硬盘` (MyDrive) 根目录
3. 确认文件路径为 `/content/drive/MyDrive/wuyin_train.jsonl`

### 2. 打开 Colab Notebook

1. 打开 [Google Colab](https://colab.research.google.com)
2. 点击 `文件` -> `上传笔记本`，选择 `train_wuyin.ipynb`
3. 或直接从 Google Drive 打开 `.ipynb` 文件

### 3. 配置运行时

在 Colab 菜单栏:
- `运行时` -> `更改运行时类型`
- 硬件加速器: **T4 GPU** (免费版即可)
- 点击保存

### 4. 按顺序执行单元格

1. **Cell 1**: 检查 GPU — 确认看到 T4 或 V100
2. **Cell 2**: 挂载 Google Drive — 点击链接授权
3. **Cell 3**: 安装依赖 — 约 2-3 分钟
4. **Cell 4**: 导入和配置 — 可根据需要修改参数
5. **Cell 5**: 加载模型 — 首次下载约 3-5 分钟
6. **Cell 6**: 配置 LoRA — 打印可训练参数数量
7. **Cell 7**: 加载数据 — 验证数据格式
8. **Cell 8**: 开始训练 — 观察 loss 下降
9. **Cell 9**: 保存到 Drive — adapter 保存到 Google Drive
10. **Cell 10**: 推理测试 (可选) — 验证训练效果

### 5. Colab 注意事项

- **会话超时**: 免费版 Colab 约 2-4 小时后断开，训练完成后会保存到 Drive，数据不会丢失
- **断点续训**: 如需从 checkpoint 继续，修改 Cell 4 中的输出目录指向 checkpoint 位置
- **防止断开**: 可以安装浏览器插件保持 Colab 连接活跃
- **监控显存**: 在 Cell 1 的 `nvidia-smi` 可以查看显存使用情况

---

## 合并 Adapter 与基础模型

训练完成后，需要将 LoRA adapter 合并回基础模型才能正常部署。

### 方法 1: 使用 PEFT merge_and_unload (推荐)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "wuyin-lora-adapter"
MERGED_PATH = "wuyin-merged"

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载并合并 adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()

# 保存合并后的完整模型
model.save_pretrained(MERGED_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_PATH)

print(f"合并完成! 模型保存到: {MERGED_PATH}")
```

将以上代码保存为 `merge_adapter.py` 并运行:
```bash
python merge_adapter.py
```

### 方法 2: 使用 mergekit 工具

```bash
pip install mergekit
```

然后使用 mergekit 的 LoRA 合并功能。

### 合并后模型大小

| 状态 | 大小 |
|------|------|
| 基础模型 (FP16) | ~3 GB |
| LoRA Adapter | ~10-50 MB |
| 合并后 (FP16) | ~3 GB |

---

## 推理测试

### 加载合并后模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "wuyin-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("wuyin-merged", trust_remote_code=True)

messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 仅加载 Adapter (不合并)

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, "wuyin-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("wuyin-lora-adapter", trust_remote_code=True)

# 推理代码同上
```

---

## 常见错误排查

### 1. CUDA Out of Memory (OOM)

**错误信息**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**解决方案**:
- 减小 `BATCH_SIZE` 从 2 到 1
- 增大 `GRAD_ACCUM_STEPS` 从 4 到 8
- 减小 `MAX_SEQ_LENGTH` 从 2048 到 1024
- 关闭其他占用 GPU 的程序
- 使用 `nvidia-smi` 查看显存占用

### 2. bitsandbytes 安装失败

**Windows**:
```bash
# 下载预编译的 wheel
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

**Linux**:
```bash
# 确保 CUDA Toolkit 已安装
nvcc --version
pip install bitsandbytes --upgrade
```

### 3. 数据格式错误

**错误信息**: `KeyError: 'messages'` 或 `JSONDecodeError`

**解决方案**:
- 检查 JSONL 文件每行是否是独立有效的 JSON
- 确认每条数据包含 `messages` 字段
- 用以下 Python 代码验证:
```python
import json
with open("wuyin_train.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            assert "messages" in data, f"第 {i} 行缺少 messages 字段"
        except Exception as e:
            print(f"第 {i} 行错误: {e}")
```

### 4. flash-attention 安装失败

**说明**: flash-attention 是可选的，不影响训练功能。

**解决方案**:
- 在 `train_wuyin.py` 中删除或注释 `attn_implementation="flash_attention_2"` 这行
- 模型会自动降级为 SDPA (PyTorch 原生实现)

### 5. 训练 loss 不下降

**可能原因**:
- 学习率过高或过低 — 尝试 1e-4 到 5e-4
- 数据量太少 — 至少需要 50-100 条数据
- 数据格式不对 — 检查 chat template 是否正确
- 尝试使用 Instruct 版本: `Qwen/Qwen2.5-1.5B-Instruct`

### 6. bf16 不支持

**错误信息**: `RuntimeError: BFloat16 is not supported on this GPU`

**解决方案**:
- 在 TrainingArguments 中将 `bf16=True` 改为 `fp16=True`
- 仅 GTX 10xx / 16xx 系列会出现此问题，RTX 20xx+ 均支持 bf16

### 7. 模型下载慢或失败

**解决方案**:
```bash
# 使用 HF 镜像站
export HF_ENDPOINT=https://hf-mirror.com
python train_wuyin.py

# 或在 Python 代码中设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `train_wuyin.py` | 本地 QLoRA 训练脚本 |
| `train_wuyin.ipynb` | Google Colab 交互式 Notebook |
| `requirements.txt` | Python 依赖列表 |
| `README_TRAINING.md` | 本文档 |
| `wuyin_train.jsonl` | 训练数据（需自行准备） |
| `wuyin-lora-adapter/` | 训练输出目录（训练后生成） |
| `wuyin-merged/` | 合并后完整模型目录（合并后生成） |

---

## 快速开始 (TL;DR)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（将你的 JSONL 数据放到当前目录）
# wuyin_train.jsonl

# 3. 训练
python train_wuyin.py

# 4. 合并模型
python merge_adapter.py

# 5. 推理测试
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained('wuyin-merged', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('wuyin-merged', trust_remote_code=True)
msg = [{'role':'user','content':'你好'}]
text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
out = model.generate(**tokenizer(text, return_tensors='pt').to('cuda'), max_new_tokens=256)
print(tokenizer.decode(out[0], skip_special_tokens=True))
"
```
