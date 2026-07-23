# 无垠AI 集成指南

将 无垠AI 替换为任何项目的 AI 对话后端。调用方式与 OpenAI Chat Completions API 完全兼容。

---

## 1. 星辰校园论坛 (Spring Boot)

**文件位置**: `DeepSeekService.java`（通常在 `com.starry.service` 包下）

### 方案 A：仅改 1 行 (如果已有 OpenAI 兼容 Client)

```java
// 修改前：
private static final String API_URL = "https://api.deepseek.com/v1/chat/completions";

// 修改后：
private static final String API_URL = "http://localhost:11434/v1/chat/completions";
```

同时修改 model 名称：

```java
// 修改前：
private static final String MODEL = "deepseek-chat";

// 修改后：
private static final String MODEL = "wuyin-ai";
```

### 方案 B：切换为 Python 微服务 (推荐，无需改 Java)

1. 安装依赖并启动 Python 桥接服务：

```bash
pip install flask requests sseclient-py
python wuyin_bridge.py   # 监听 5000 端口，完全模拟 DeepSeek API 格式
```

2. Java 端只改 1 行 URL：

```java
private static final String API_URL = "http://localhost:5000/v1/chat/completions";
```

### 方案 C：直接注入 WuyinClient Bean

如果你的项目使用 Spring Boot，可以创建一个 `WuyinService.java` 替代 `DeepSeekService.java`：

```java
@Service
public class WuyinService {
    private final RestTemplate restTemplate = new RestTemplate();
    private static final String URL = "http://localhost:11434/v1/chat/completions";

    public String chat(List<Message> messages) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        Map<String, Object> body = Map.of(
            "model", "wuyin-ai",
            "messages", messages
        );
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        ResponseEntity<Map> resp = restTemplate.postForEntity(URL, request, Map.class);
        Map choice = (Map) ((List) resp.getBody().get("choices")).get(0);
        return (String) ((Map) choice.get("message")).get("content");
    }
}
```

---

## 2. 便民服务系统 (Express/Node.js)

**替换 DeepSeek API 调用为本地 Ollama**：

### 修改前 (调用 DeepSeek):

```javascript
const response = await fetch("https://api.deepseek.com/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${process.env.DEEPSEEK_API_KEY}`
  },
  body: JSON.stringify({
    model: "deepseek-chat",
    messages: messages
  })
});
```

### 修改后 (调用无垠AI，仅改 URL 和 model):

```javascript
const response = await fetch("http://localhost:11434/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer ollama"  // Ollama 不需要真实 key
  },
  body: JSON.stringify({
    model: "wuyin-ai",
    messages: messages
  })
});
```

### Express 中间件封装:

```javascript
// middleware/wuyin.js
const WUYIN_URL = process.env.WUYIN_URL || "http://localhost:11434/v1/chat/completions";

async function wuyinChat(messages) {
  const resp = await fetch(WUYIN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": "Bearer ollama" },
    body: JSON.stringify({ model: "wuyin-ai", messages })
  });
  const data = await resp.json();
  return data.choices[0].message.content;
}

// 在路由中使用:
app.post("/api/ai/chat", async (req, res) => {
  const reply = await wuyinChat(req.body.messages);
  res.json({ reply });
});
```

---

## 3. 任意项目通用集成

无垠AI 提供标准 OpenAI Chat Completions 接口，以下语言都可以直接调用：

### cURL

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d '{
    "model": "wuyin-ai",
    "messages": [
      {"role": "system", "content": "你是无垠AI，校园知识智能助手。"},
      {"role": "user", "content": "介绍一下学校的图书馆"}
    ]
  }'
```

### Python (官方 OpenAI SDK 兼容)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama 不需要真实 key
)

response = client.chat.completions.create(
    model="wuyin-ai",
    messages=[
        {"role": "system", "content": "你是无垠AI，校园知识智能助手。"},
        {"role": "user", "content": "介绍一下学校的图书馆"},
    ],
)

print(response.choices[0].message.content)
```

### Python (wuyin_client.py，直接使用本项目客户端)

```python
from wuyin_client import WuyinClient

client = WuyinClient()  # 默认连接 localhost:11434

# 单轮对话
reply = client.chat("你好，无垠AI！")

# 多轮对话
history = []
reply1 = client.chat("我叫小明", history)
history.append({"role": "user", "content": "我叫小明"})
history.append({"role": "assistant", "content": reply1})
reply2 = client.chat("我叫什么名字？", history)

# 流式对话
for chunk in client.chat_stream("讲个笑话"):
    print(chunk, end="")
```

### JavaScript/TypeScript

```typescript
const response = await fetch("http://localhost:11434/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer ollama"
  },
  body: JSON.stringify({
    model: "wuyin-ai",
    messages: [
      { role: "system", content: "你是无垠AI，校园知识智能助手。" },
      { role: "user", content: "你好" }
    ]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### Go

```go
import (
    "bytes"
    "encoding/json"
    "net/http"
)

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type ChatRequest struct {
    Model    string    `json:"model"`
    Messages []Message `json:"messages"`
}

func wuyinChat(userMessage string, history []Message) (string, error) {
    messages := append([]Message{{Role: "system", Content: "你是无垠AI"}}, history...)
    messages = append(messages, Message{Role: "user", Content: userMessage})

    body, _ := json.Marshal(ChatRequest{Model: "wuyin-ai", Messages: messages})
    resp, err := http.Post(
        "http://localhost:11434/v1/chat/completions",
        "application/json",
        bytes.NewReader(body),
    )
    if err != nil { return "", err }
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    choices := result["choices"].([]interface{})
    choice := choices[0].(map[string]interface{})
    message := choice["message"].(map[string]interface{})
    return message["content"].(string), nil
}
```

### Rust

```rust
use serde_json::json;
use reqwest::Client;

async fn wuyin_chat(message: &str) -> Result<String, reqwest::Error> {
    let client = Client::new();
    let resp = client
        .post("http://localhost:11434/v1/chat/completions")
        .header("Authorization", "Bearer ollama")
        .json(&json!({
            "model": "wuyin-ai",
            "messages": [
                {"role": "system", "content": "你是无垠AI，校园知识智能助手。"},
                {"role": "user", "content": message}
            ]
        }))
        .send()
        .await?;

    let data: serde_json::Value = resp.json().await?;
    Ok(data["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
}
```

---

## 关键差异：从 DeepSeek API 迁移到无垠AI

| 项目 | DeepSeek API | 无垠AI (Ollama) |
|------|-------------|-----------------|
| URL | `https://api.deepseek.com/v1/chat/completions` | `http://localhost:11434/v1/chat/completions` |
| API Key | 真实 key | `ollama` (任意值) |
| Model | `deepseek-chat` | `wuyin-ai` |
| 费用 | 按 token 计费 | **免费** |
| 网络 | 需要外网 | **纯离线/内网** |
| 隐私 | 数据上传云端 | **数据不出服务器** |

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WUYIN_BASE_URL` | `http://localhost:11434/v1` | API 地址 |
| `WUYIN_MODEL` | `wuyin-ai` | 模型名称 |
| `WUYIN_SYSTEM` | 无垠AI 系统提示词 | 系统提示词 |
| `WUYIN_TIMEOUT` | `60` | 请求超时 (秒) |
| `WUYIN_MAX_TOKENS` | `2048` | 最大生成 tokens |
