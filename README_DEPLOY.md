# 无垠AI 部署指南

> 基于 Qwen2.5-1.5B + LoRA 微调的校园知识智能助手，通过 Ollama 本地部署，数据不出服务器。

---

## 目录

1. [系统要求](#系统要求)
2. [环境准备](#环境准备)
3. [模型准备](#模型准备)
4. [LoRA 微调 (可选)](#lora-微调-可选)
5. [一键部署](#一键部署)
6. [手动部署步骤](#手动部署步骤)
7. [验证部署](#验证部署)
8. [API 调用方式](#api-调用方式)
9. [集成到现有项目](#集成到现有项目)
10. [常见问题](#常见问题)
11. [性能调优](#性能调优)

---

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 10 GB 可用 | 20 GB SSD |
| GPU (可选) | NVIDIA 4GB VRAM | NVIDIA 8GB+ VRAM (CUDA 12+) |
| 操作系统 | Windows 10+ / Linux / macOS | Windows 11 / Ubuntu 22.04 |
| Python | 3.10+ | 3.11+ |
| 网络 | 无需外网 (纯离线运行) | — |

---

## 环境准备

### 1. 安装 Python 依赖

```bash
pip install transformers peft torch accelerate
pip install requests sseclient-py
```

### 2. 安装 Ollama

**Windows:**
从 https://ollama.com/download/windows 下载安装包，双击安装。

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
从 https://ollama.com/download/mac 下载安装包。

### 3. 克隆 llama.cpp (用于 GGUF 转换)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

### 4. 验证安装

```bash
ollama --version
python -c "import transformers; print(transformers.__version__)"
python -c "import peft; print(peft.__version__)"
```

---

## 模型准备

### 方案 A：使用预合并模型 (推荐)

如果你已经有合并好的 GGUF 文件，直接放到 `E:\claude code\wuyin-ai\` 目录：

```
wuyin-ai/
  Modelfile
  wuyin-qwen2.5-1.5b-merged.Q4_K_M.gguf   <-- 放这里
  deploy_wuyin.bat
  wuyin_client.py
  ...
```

然后执行：

```bash
ollama create wuyin-ai -f Modelfile
```

### 方案 B：从 HuggingFace/Ollama 拉取基础模型

```bash
# 拉取 Qwen2.5-1.5B
ollama pull qwen2.5:1.5b

# 或者从 HuggingFace 下载原始权重
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir E:\models\Qwen2.5-1.5B
```

---

## LoRA 微调 (可选)

如果你的 LoRA adapter 还未训练，参考以下流程。

### 1. 准备训练数据

数据格式 (JSONL)：

```json
{"instruction": "介绍一下学校的图书馆", "output": "我们学校的图书馆位于..."}
{"instruction": "怎么加入计算机社团？", "output": "加入计算机社团的流程是..."}
```

### 2. 训练 LoRA Adapter

```bash
python train_lora.py \
  --model_name_or_path E:\models\Qwen2.5-1.5B \
  --data_path E:\data\campus_qa.jsonl \
  --output_dir E:\models\wuyin-lora-adapter \
  --lora_r 8 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-4 \
  --save_steps 500 \
  --fp16
```

### 3. 验证 LoRA 效果

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "E:/models/Qwen2.5-1.5B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "E:/models/wuyin-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("E:/models/Qwen2.5-1.5B")

inputs = tokenizer("你好，介绍一下学校", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 一键部署

**Windows:**

```bat
cd /d "E:\claude code\wuyin-ai"
deploy_wuyin.bat
```

**Linux/macOS:**

```bash
cd /path/to/wuyin-ai
chmod +x deploy_wuyin.sh
./deploy_wuyin.sh
```

一键部署脚本会自动完成：
1. 合并 LoRA adapter 到基础模型
2. 转换为 GGUF Q4_K_M 格式
3. 创建 Ollama model
4. 启动 Ollama serve
5. 执行测试查询

---

## 手动部署步骤

如果你需要手动控制每个步骤：

### Step 1: 合并 LoRA 到基础模型

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_dir = 'E:/models/Qwen2.5-1.5B'
lora_dir = 'E:/models/wuyin-lora-adapter'
output_dir = 'E:/models/wuyin-merged'

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto'
)

print('Loading LoRA adapter...')
model = PeftModel.from_pretrained(base, lora_dir)

print('Merging...')
merged = model.merge_and_unload()

print('Saving merged model...')
merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size='5GB')
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f'Done! Merged model at: {output_dir}')
"
```

### Step 2: 转换为 GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py \
  E:/models/wuyin-merged \
  --outtype q4_k_m \
  --outfile "E:/claude code/wuyin-ai/wuyin-qwen2.5-1.5b-merged.Q4_K_M.gguf"
```

**GGUF 量化等级说明：**

| 格式 | 大小 (1.5B) | 质量 | 适用 |
|------|------------|------|------|
| Q2_K | ~500 MB | 低 | 极低资源 |
| Q4_K_M | ~1 GB | 中高 | **推荐 (本部署)** |
| Q5_K_M | ~1.3 GB | 高 | 高质量需求 |
| Q8_0 | ~1.8 GB | 极高 | GPU 充足 |
| F16 | ~3 GB | 无损 | 不推荐本地 |

### Step 3: 创建 Ollama 模型

```bash
cd "E:\claude code\wuyin-ai"
ollama create wuyin-ai -f Modelfile
```

验证模型列表：

```bash
ollama list
# 应该看到: wuyin-ai:latest  ...
```

### Step 4: 启动服务

```bash
# 启动 Ollama (如果未运行)
ollama serve

# 或在后台运行
start ollama serve          # Windows
ollama serve &              # Linux/macOS
```

### Step 5: 测试模型

```bash
# 命令行交互式测试
ollama run wuyin-ai

# API 测试
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d "{\"model\":\"wuyin-ai\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，介绍一下你自己\"}]}"
```

---

## 验证部署

### 1. 健康检查

```bash
curl http://localhost:11434/api/tags
```

预期返回包含 `wuyin-ai` 的模型列表。

### 2. Python 客户端测试

```bash
cd "E:\claude code\wuyin-ai"
python wuyin_client.py
```

在交互界面输入测试问题。

### 3. 接口兼容性测试

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="wuyin-ai",
    messages=[
        {"role": "system", "content": "你是无垠AI，校园知识智能助手。"},
        {"role": "user", "content": "学校有哪些社团？"},
    ],
    temperature=0.7,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

### 4. 异常测试

```bash
# 测试空消息
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d "{\"model\":\"wuyin-ai\",\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}"
```

---

## API 调用方式

### OpenAI 兼容 API (推荐)

```
POST http://localhost:11434/v1/chat/completions
```

完全兼容 OpenAI SDK、LangChain、以及其他 OpenAI-format 客户端。

### Ollama 原生 API

```
POST http://localhost:11434/api/generate
POST http://localhost:11434/api/chat
```

详见 [Ollama API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md)。

### 流式输出

```python
from wuyin_client import WuyinClient

client = WuyinClient()
for chunk in client.chat_stream("讲一个校园故事"):
    print(chunk, end="", flush=True)
```

---

## 集成到现有项目

### 星辰校园论坛 (Spring Boot)

修改 `DeepSeekService.java`：

```java
// 修改这两处即可
private static final String API_URL = "http://localhost:11434/v1/chat/completions";
private static final String MODEL = "wuyin-ai";
// API Key 可以设为任意值 (如 "ollama")
```

### 便民服务系统 (Express)

```javascript
const WUYIN_URL = "http://localhost:11434/v1/chat/completions";

async function wuyinChat(messages) {
  const resp = await fetch(WUYIN_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer ollama"
    },
    body: JSON.stringify({ model: "wuyin-ai", messages })
  });
  const data = await resp.json();
  return data.choices[0].message.content;
}
```

更多集成方式详见 [integration_guide.md](./integration_guide.md)。

---

## 常见问题

### Q: "ollama create" 报找不到 Modelfile？

确保在 `wuyin-ai` 目录下执行，或使用绝对路径：
```bash
ollama create wuyin-ai -f "E:\claude code\wuyin-ai\Modelfile"
```

### Q: GGUF 转换报内存不足？

减小转换批次或使用更低的量化等级：
```bash
python convert_hf_to_gguf.py E:/models/wuyin-merged --outtype q4_0 --outfile model.gguf
```

### Q: Ollama 启动后无法连接？

1. 检查端口：`netstat -an | findstr 11434` (Windows) / `lsof -i :11434` (Linux)
2. 检查防火墙是否放行 11434 端口
3. 设置环境变量 `OLLAMA_HOST=0.0.0.0:11434` 以允许远程连接

### Q: 回复速度太慢？

1. 启用 GPU：`ollama serve` 会自动检测 CUDA
2. 减小 `num_ctx` 参数 (Modelfile 中默认 4096)
3. 使用更低量化等级 (Q4_K_M 已经平衡了速度与质量)

### Q: 如何让其他机器访问？

```bash
# 设置 Ollama 监听所有网络接口
set OLLAMA_HOST=0.0.0.0:11434    # Windows
export OLLAMA_HOST=0.0.0.0:11434 # Linux/macOS
ollama serve
```

### Q: 如何更新模型？

```bash
# 重新创建即可覆盖
ollama create wuyin-ai -f Modelfile

# 删除旧模型
ollama rm wuyin-ai
```

### Q: 如何备份模型？

```bash
# 导出为 tar
ollama save wuyin-ai -o wuyin-ai-backup.tar

# 恢复
ollama load wuyin-ai -i wuyin-ai-backup.tar
```

---

## 性能调优

### Modelfile 参数调优

```
# 推理速度优先
PARAMETER num_ctx 2048          # 减小上下文窗口
PARAMETER num_predict 512       # 限制生成长度

# 质量优先
PARAMETER num_ctx 8192          # 更大上下文
PARAMETER temperature 0.7       # 适中创造性
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1    # 减少重复

# 确定性输出 (评测用)
PARAMETER temperature 0
PARAMETER top_k 1
```

### GPU 加速

Ollama 自动检测 NVIDIA GPU (CUDA)。确认 GPU 是否被使用：

```bash
ollama run wuyin-ai --verbose
# 查看输出中的 "eval rate" — GPU 模式通常 >20 tokens/s
```

### 并发服务

对于生产环境，建议：
1. 使用 `systemd` (Linux) 或 NSSM (Windows) 将 Ollama 注册为系统服务
2. 在前面加 Nginx 反向代理实现负载均衡
3. 使用 `OLLAMA_NUM_PARALLEL` 环境变量控制并发数

```bash
export OLLAMA_NUM_PARALLEL=4    # 最多 4 个并发请求
export OLLAMA_MAX_LOADED_MODELS=2
```

---

## 文件清单

```
wuyin-ai/
  Modelfile                           # Ollama 模型定义
  deploy_wuyin.bat                    # Windows 一键部署脚本
  wuyin_client.py                     # Python 客户端 (OpenAI 接口兼容)
  integration_guide.md                # 多项目集成指南
  README_DEPLOY.md                    # 本文件
  wuyin-qwen2.5-1.5b-merged.Q4_K_M.gguf  # (部署后生成) 合并后的 GGUF 模型
```

---

## 联系我们

- 项目仓库: (待填写)
- 模型基于: Qwen2.5-1.5B (Apache 2.0 License)
- Ollama: https://ollama.com
- llama.cpp: https://github.com/ggerganov/llama.cpp
