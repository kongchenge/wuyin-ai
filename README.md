# 无垠AI - 训练数据生成工具

从 `campus_forum.knowledge_base` 表读取校园知识库文档，使用 DeepSeek API 批量生成中文问答对，输出 OpenAI chat 格式 JSONL 文件，用于模型微调。

## 快速开始

### 1. 安装依赖

```bash
pip install mysql-connector-python requests tqdm
```

### 2. 确保数据库可访问

MySQL 连接配置（脚本内置）：

| 参数     | 值              |
| -------- | --------------- |
| host     | `localhost`     |
| user     | `root`          |
| password | `wuyi12345`     |
| database | `campus_forum`  |

### 3. 运行

```bash
# 正式运行（自动断点续传）
python generate_training_data.py

# 从头开始（忽略已有 checkpoint）
python generate_training_data.py --fresh

# 预览模式（不调用 API）
python generate_training_data.py --dry-run

# 调试模式（只处理前 5 篇）
python generate_training_data.py --max-docs 5
```

## 工作原理

```
knowledge_base 表 (81 篇)
       │
       ▼
  按文档长度自适应 QA 数量
  ├─ < 200 字  → 15 对
  ├─ < 500 字  → 25 对
  ├─ < 1000 字 → 40 对
  ├─ < 2000 字 → 50 对
  └─ >=2000 字 → 60 对
       │
       ▼
  分批调用 DeepSeek API（每批 10 对）
  速率: 1 req/s, 超时: 120s, 重试: 3 次
       │
       ▼
  wuyin_train.jsonl (≈4000 QA 对)
```

## 输出格式

每行一条 OpenAI chat 格式 JSON：

```json
{
  "messages": [
    {"role": "system", "content": "你是无垠AI，校园知识助手"},
    {"role": "user", "content": "一卡通丢失了怎么办？"},
    {"role": "assistant", "content": "发现一卡通丢失后请立即登录校园门户网站或拨打热线8973-1234进行挂失。补办需要携带身份证和学生证到行政楼102室，工本费20元。"}
  ]
}
```

## 特性

| 特性       | 说明                                           |
| ---------- | ---------------------------------------------- |
| 断点续传   | `checkpoint.json` 记录进度，中断后可继续        |
| 速率限制   | 请求间隔 1 秒，避免 API 限流                   |
| 自动重试   | 失败后指数退避重试（最多 3 次）                 |
| 鲁棒解析   | 自动处理 markdown 代码块、JSON 截断等异常输出   |
| 格式验证   | 生成完成后自动校验每条数据的 role/content 完整性 |
| 日志       | 控制台 + 文件双输出（`generate_training_data.log`） |

## 预估耗时

- 81 篇文档，约 4000 QA 对
- API 调用次数：约 400 次（分批）
- 速率 1 req/s：约 **7-10 分钟**

## 文件说明

| 文件                           | 用途                       |
| ------------------------------ | -------------------------- |
| `generate_training_data.py`    | 主脚本                     |
| `wuyin_train.jsonl`            | 生成的训练数据（输出）      |
| `checkpoint.json`              | 断点进度（运行中，完成后自动删除） |
| `generate_training_data.log`   | 运行日志                   |
