"""
无垠AI Python Client — 兼容 Ollama OpenAI-compatible API

接口对标 星辰校园论坛 DeepSeekService.java:
  - chat(message, history)        → 多轮对话
  - chatStream(message, history)  → 流式对话 (SSE generator)
  - 配置方式与 DeepSeekService 一致：构造函数或环境变量

Usage:
    from wuyin_client import WuyinClient

    client = WuyinClient()                        # 默认 localhost:11434
    client = WuyinClient(base_url="...", model="wuyin-ai")

    reply = client.chat("你好")
    for chunk in client.chat_stream("你好"):
        print(chunk, end="")
"""

import os
import json
import logging
from typing import Optional, Generator

import requests  # pip install requests
import sseclient  # pip install sseclient-py

logger = logging.getLogger("wuyin-client")


class WuyinClient:
    """Ollama 通用客户端，接口与 DeepSeekService 完全兼容。

    对标 Java 版 DeepSeekService：
      - chat(String message, List<Message> history)           → chat(message, history)
      - chatStream(String message, List<Message> history)     → chat_stream(message, history)
    """

    # --- 默认值 ---
    _DEFAULT_BASE_URL = os.environ.get("WUYIN_BASE_URL", "http://localhost:11434/v1")
    _DEFAULT_MODEL = os.environ.get("WUYIN_MODEL", "wuyin-ai")
    _DEFAULT_SYSTEM = os.environ.get(
        "WUYIN_SYSTEM",
        "你是无垠AI，一个专注于校园知识的智能助手。你了解校园生活、学习交流、技术讨论、社团活动等方面的知识。"
    )
    _DEFAULT_TIMEOUT = int(os.environ.get("WUYIN_TIMEOUT", "60"))
    _DEFAULT_MAX_TOKENS = int(os.environ.get("WUYIN_MAX_TOKENS", "2048"))

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        """初始化 WuyinClient。

        参数命名保持与 DeepSeekService 构造函数一致：
          - base_url:    Ollama API endpoint (默认 http://localhost:11434/v1)
          - model:       模型名称 (默认 wuyin-ai)
          - system_prompt: 系统提示词
          - api_key:     Ollama 默认不需要，保留以兼容接口
          - timeout:     请求超时秒数
          - max_tokens:  最大生成 tokens
        """
        self.base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self._DEFAULT_MODEL
        self.system_prompt = system_prompt or self._DEFAULT_SYSTEM
        self.api_key = api_key or "ollama"  # Ollama 不需要真实 key，但 header 要有
        self.timeout = timeout or self._DEFAULT_TIMEOUT
        self.max_tokens = max_tokens or self._DEFAULT_MAX_TOKENS

        self._chat_url = f"{self.base_url}/chat/completions"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info("WuyinClient initialized: model=%s url=%s", self.model, self._chat_url)

    # ----------------------------------------------------------------
    #  Public API — 与 DeepSeekService.chat / chatStream 完全一致
    # ----------------------------------------------------------------

    def chat(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """非流式多轮对话。

        Args:
            message:    用户当前消息
            history:    历史对话 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
            temperature: 生成温度 (默认使用 Modelfile 设置)
            max_tokens:  最大 tokens

        Returns:
            模型回复文本。失败时返回空字符串。
        """
        messages = self._build_messages(message, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            resp = requests.post(
                self._chat_url,
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error("chat request failed: %s", e)
            return ""
        except (KeyError, IndexError, TypeError) as e:
            logger.error("chat response parse error: %s", e)
            return ""

    def chat_stream(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """流式多轮对话 (SSE generator)。

        用法与 DeepSeekService.chatStream 一致：
            for chunk in client.chat_stream("你好"):
                print(chunk, end="")

        Args:
            message:    用户当前消息
            history:    历史对话
            temperature: 生成温度
            max_tokens:  最大 tokens

        Yields:
            每次 yield 一段 delta 文本。
        """
        messages = self._build_messages(message, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            resp = requests.post(
                self._chat_url,
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()

            # Ollama v1/chat/completions 返回标准 SSE
            client = sseclient.SSEClient(resp)
            for event in client.events():
                if event.data == "[DONE]":
                    break
                try:
                    data = json.loads(event.data)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        except requests.RequestException as e:
            logger.error("chat_stream request failed: %s", e)
        except Exception as e:
            logger.error("chat_stream error: %s", e)

    # ----------------------------------------------------------------
    #  Internal helpers
    # ----------------------------------------------------------------

    def _build_messages(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
    ) -> list[dict[str, str]]:
        """构建 messages 列表，始终在最前面插入 system prompt。"""
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            for h in history:
                role = h.get("role", "user")
                content = h.get("content", "")
                if role in ("user", "assistant", "system"):
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    def health_check(self) -> bool:
        """检查 Ollama 服务是否可达。"""
        try:
            resp = requests.get(
                f"{self.base_url.rsplit('/', 1)[0]}/../api/tags",
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ----------------------------------------------------------------
#  便捷函数 — 对标 Spring Boot @Bean 单例模式
# ----------------------------------------------------------------
_client_instance: Optional[WuyinClient] = None


def get_client() -> WuyinClient:
    """获取全局单例 (对标 Spring @Bean)。"""
    global _client_instance
    if _client_instance is None:
        _client_instance = WuyinClient()
    return _client_instance


# ----------------------------------------------------------------
#  CLI 测试入口
# ----------------------------------------------------------------
if __name__ == "__main__":
    import sys

    client = WuyinClient()

    # 健康检查
    if not client.health_check():
        print("[ERROR] Ollama 服务未运行，请先执行: ollama serve")
        sys.exit(1)

    print(f"无垠AI Client ready.  model={client.model}  url={client._chat_url}")
    print("输入消息开始对话，输入 /exit 退出，输入 /stream 切换流式模式。\n")

    use_stream = False
    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break
        if user_input == "/stream":
            use_stream = not use_stream
            print(f"[stream mode: {'ON' if use_stream else 'OFF'}]")
            continue

        if use_stream:
            print("Wuyin> ", end="", flush=True)
            full = ""
            for chunk in client.chat_stream(user_input, history):
                print(chunk, end="", flush=True)
                full += chunk
            print()
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full})
        else:
            reply = client.chat(user_input, history)
            print(f"Wuyin> {reply}")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
