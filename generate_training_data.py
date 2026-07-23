#!/usr/bin/env python3
"""
generate_training_data.py
==========================
从 campus_forum.knowledge_base 表读取知识库文档，使用 DeepSeek API 批量生成
中文问答对，输出 OpenAI chat 格式 JSONL 用于模型微调。

特性:
  - 自适应 QA 数量（按文档长度缩放，总目标 ~4000 对）
  - 分批生成（每批 10 对）保证多样性
  - 速率限制（1 req/s）、指数退避重试（最多 3 次）
  - 断点续传（checkpoint.json 记录已完成文档 ID）
  - tqdm 进度条 + 结构化日志

用法:
  python generate_training_data.py [--resume] [--dry-run]

依赖:
  pip install mysql-connector-python requests tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG: Dict[str, Any] = {
    "host": "localhost",
    "user": "root",
    "password": "wuyi12345",
    "database": "campus_forum",
    "charset": "utf8mb4",
    "autocommit": True,
}

DEEPSEEK_API_KEY: str = "sk-931b079c6210423a852f9f16c9aa4ec4"
DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL: str = "deepseek-chat"

QA_PER_DOC_DEFAULT: int = 50          # 默认每文档 QA 对数
BATCH_SIZE: int = 10                   # 每批 API 调用生成的 QA 数
RATE_LIMIT_SECONDS: float = 1.0        # 请求间隔（秒）
MAX_RETRIES: int = 3                   # 最大重试次数
RETRY_BACKOFF: float = 2.0             # 退避乘数
REQUEST_TIMEOUT: int = 120             # API 请求超时（秒）

SYSTEM_PROMPT: str = "你是无垠AI，校园知识助手"

BASE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_FILE: Path = BASE_DIR / "wuyin_train.jsonl"
CHECKPOINT_FILE: Path = BASE_DIR / "checkpoint.json"
LOG_FILE: Path = BASE_DIR / "generate_training_data.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("generate_training_data")
logger.setLevel(logging.DEBUG)

# 控制台 handler（INFO 级别）
_console = logging.StreamHandler(sys.stderr)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_console)

# 文件 handler（DEBUG 级别）
_file = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(_file)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qa_target_for_length(content_length: int) -> int:
    """根据文档长度自适应决定 QA 数量，使总数接近 4000。"""
    if content_length < 200:
        return 15
    elif content_length < 500:
        return 25
    elif content_length < 1000:
        return 40
    elif content_length < 2000:
        return 50
    else:
        return 60  # 长文档稍多补短文档缺口


def _build_generation_prompt(title: str, category: str, content: str, count: int) -> str:
    """构建发给 DeepSeek 的生成 prompt。"""
    return textwrap.dedent(f"""\
        你是一个专业的数据标注助手。请根据以下知识库文档，生成 {count} 个多样化的中文问答对。

        【文档标题】{title}
        【文档分类】{category or "未分类"}

        【文档内容】
        {content}

        【要求】
        1. 问题类型多样化：是什么、为什么、怎么做、优缺点、对比、举例等。
        2. 答案必须严格基于文档内容，不能胡编。
        3. 问题和答案都用中文，表达自然。
        4. 以纯 JSON 数组格式输出，不要加 markdown 代码块标记，不要输出其他文字。
        5. 格式：[{{"question": "问题", "answer": "答案"}}, ...]

        现在请输出 {count} 个问答对：""")


def _parse_qa_json(text: str) -> List[Dict[str, str]]:
    """从 API 响应中鲁棒解析 JSON 问答对数组。

    处理常见情况：markdown 代码块包裹、首尾杂文、JSON 不完整。
    """
    # 1) 尝试提取 ```json ... ``` 或 ``` ... ``` 代码块
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        text = code_block.group(1)

    # 2) 找第一个 '[' 和最后一个 ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    # 3) 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                {"question": item["question"], "answer": item["answer"]}
                for item in data
                if isinstance(item, dict) and "question" in item and "answer" in item
            ]
    except json.JSONDecodeError:
        pass

    # 4) 逐行修复常见 JSON 问题后重试
    cleaned = re.sub(r",\s*]", "]", text)
    cleaned = re.sub(r",\s*}", "}", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [
                {"question": item["question"], "answer": item["answer"]}
                for item in data
                if isinstance(item, dict) and "question" in item and "answer" in item
            ]
    except json.JSONDecodeError:
        pass

    # 5) 正则兜底：提取每个 {"question": "...", "answer": "..."} 对象
    pairs: List[Dict[str, str]] = []
    for match in re.finditer(
        r'\{\s*"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        text,
    ):
        q = re.sub(r'\\(.)', r'\1', match.group(1))
        a = re.sub(r'\\(.)', r'\1', match.group(2))
        pairs.append({"question": q, "answer": a})

    return pairs


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def fetch_documents() -> List[Dict[str, Any]]:
    """从 knowledge_base 表读取全部文档。"""
    logger.info("连接数据库 %s@%s/%s ...", DB_CONFIG["user"], DB_CONFIG["host"], DB_CONFIG["database"])
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, title, content, category, uploader_id, created_at FROM knowledge_base ORDER BY id")
        rows = cursor.fetchall()
        logger.info("从 knowledge_base 读取到 %d 篇文档", len(rows))
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint() -> Tuple[set, List[Dict[str, Any]]]:
    """加载断点：返回 (已完成的文档 ID 集合, 已生成的问答对列表)。"""
    if not CHECKPOINT_FILE.exists():
        return set(), []
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        completed_ids = set(data.get("completed_doc_ids", []))
        pairs = data.get("qa_pairs", [])
        logger.info("加载断点: %d 篇已完成, %d 对已生成", len(completed_ids), len(pairs))
        return completed_ids, pairs
    except Exception as exc:
        logger.warning("断点文件损坏，将从头开始: %s", exc)
        return set(), []


def save_checkpoint(completed_ids: set, qa_pairs: List[Dict[str, Any]]) -> None:
    """保存断点。"""
    data = {
        "completed_doc_ids": sorted(completed_ids),
        "qa_pairs": qa_pairs,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    # 原子写入：先写临时文件再重命名
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# DeepSeek API
# ---------------------------------------------------------------------------

def call_deepseek(system: str, user: str, temperature: float = 0.7) -> Optional[str]:
    """调用 DeepSeek chat API，返回 assistant 文本内容；失败返回 None。"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            return content
        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning("API 超时 (attempt %d/%d)", attempt, MAX_RETRIES)
        except requests.exceptions.HTTPError as e:
            last_exc = e
            logger.warning(
                "API HTTP %s (attempt %d/%d): %s",
                resp.status_code if 'resp' in locals() else '?',
                attempt,
                MAX_RETRIES,
                e,
            )
            # 4xx 错误（非 429）不重试
            if resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 429:
                logger.error("不可重试的 HTTP %d，放弃", resp.status_code)
                return None
        except Exception as e:
            last_exc = e
            logger.warning("API 异常 (attempt %d/%d): %s", attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF ** attempt
            logger.info("等待 %.1f 秒后重试...", wait)
            time.sleep(wait)

    logger.error("API 调用失败，已达最大重试次数: %s", last_exc)
    return None


def generate_qa_batch(
    title: str, category: str, content: str, count: int
) -> List[Dict[str, str]]:
    """对一个文档生成一批（count 个）问答对。"""
    prompt = _build_generation_prompt(title, category, content, count)
    resp_text = call_deepseek(
        system="你是一个精确、可靠的数据标注助手。严格按指令输出 JSON，不要添加额外说明。",
        user=prompt,
        temperature=0.8,
    )
    if resp_text is None:
        return []

    pairs = _parse_qa_json(resp_text)
    logger.debug("  解析出 %d/%d 个问答对", len(pairs), count)
    return pairs[:count]


def generate_qa_for_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """为单个文档生成全部问答对（分批调用 API）。"""
    title = doc["title"] or "无标题"
    category = doc["category"] or ""
    content = doc["content"] or ""
    doc_id = doc["id"]

    target = _qa_target_for_length(len(content))
    num_batches = max(1, (target + BATCH_SIZE - 1) // BATCH_SIZE)

    logger.info("文档 #%d 「%s」: 目标 %d 对, %d 批次", doc_id, title, target, num_batches)

    all_pairs: List[Dict[str, Any]] = []
    for batch_idx in range(num_batches):
        remaining = target - len(all_pairs)
        batch_count = min(BATCH_SIZE, remaining)
        if batch_count <= 0:
            break

        pairs = generate_qa_batch(title, category, content, batch_count)
        if pairs:
            all_pairs.extend(pairs)

        # 速率限制
        time.sleep(RATE_LIMIT_SECONDS)

    # 格式化为 OpenAI chat 格式
    formatted: List[Dict[str, Any]] = []
    for pair in all_pairs:
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]},
            ]
        })

    logger.info("  实际生成 %d 对 (目标 %d)", len(formatted), target)
    return formatted


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(pairs: List[Dict[str, Any]], path: Path) -> None:
    """将 QA 对写入 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in pairs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("已写入 %d 条记录到 %s", len(pairs), path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_qa_pair(entry: Dict[str, Any]) -> Optional[str]:
    """验证单个 entry 格式是否正确，返回错误描述或 None。"""
    msgs = entry.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        return "messages 必须是长度为 3 的列表"
    roles = ["system", "user", "assistant"]
    for i, role in enumerate(roles):
        if msgs[i].get("role") != role:
            return f"messages[{i}].role 应该是 '{role}'"
        if not isinstance(msgs[i].get("content"), str) or not msgs[i]["content"].strip():
            return f"messages[{i}].content 为空"
    return None


def print_stats(pairs: List[Dict[str, Any]]) -> None:
    """打印数据集统计信息。"""
    if not pairs:
        logger.warning("数据集为空！")
        return

    total = len(pairs)
    q_lens = [len(p["messages"][1]["content"]) for p in pairs]
    a_lens = [len(p["messages"][2]["content"]) for p in pairs]

    logger.info("=" * 50)
    logger.info("数据集统计")
    logger.info("=" * 50)
    logger.info("  总 QA 对数:  %d", total)
    logger.info("  问题平均长度: %.0f 字 (min %d, max %d)", sum(q_lens) / total, min(q_lens), max(q_lens))
    logger.info("  答案平均长度: %.0f 字 (min %d, max %d)", sum(a_lens) / total, min(a_lens), max(a_lens))
    logger.info("  输出文件:     %s", OUTPUT_FILE)
    logger.info("  日志文件:     %s", LOG_FILE)
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="从知识库文档批量生成中文 QA 训练数据（OpenAI chat JSONL 格式）"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从 checkpoint 恢复中断的运行（默认自动检测）",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="忽略 checkpoint，从头开始",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只统计文档数量不调用 API",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="最多处理 N 篇文档（调试用）",
    )
    args = parser.parse_args()

    logger.info("========== generate_training_data.py 启动 ==========")
    logger.info("输出文件: %s", OUTPUT_FILE)

    # ---- 读取文档 ----
    documents = fetch_documents()
    if not documents:
        logger.error("knowledge_base 表为空，退出")
        sys.exit(1)

    if args.dry_run:
        total_target = sum(_qa_target_for_length(len(d.get("content", ""))) for d in documents)
        logger.info("[dry-run] 共 %d 篇文档, 预估 QA 总数 ~%d", len(documents), total_target)
        return

    if args.max_docs:
        documents = documents[: args.max_docs]
        logger.info("限制处理前 %d 篇文档", args.max_docs)

    # ---- 断点恢复 ----
    all_pairs: List[Dict[str, Any]] = []
    completed_ids: set = set()

    if not args.fresh:
        completed_ids, all_pairs = load_checkpoint()

    # ---- 生成 QA 对 ----
    pending_docs = [d for d in documents if d["id"] not in completed_ids]
    total_estimate = sum(_qa_target_for_length(len(d.get("content", ""))) for d in pending_docs)

    logger.info(
        "待处理: %d 篇 (已完成 %d), 预估新增 QA: ~%d",
        len(pending_docs),
        len(completed_ids),
        total_estimate,
    )

    if not pending_docs:
        logger.info("所有文档已处理完毕！")
        write_jsonl(all_pairs, OUTPUT_FILE)
        print_stats(all_pairs)
        return

    # 进度条：按文档计数
    with tqdm(
        total=len(pending_docs),
        desc="生成 QA",
        unit="doc",
        ncols=100,
    ) as pbar:
        for doc in pending_docs:
            doc_id = doc["id"]
            title = doc["title"] or "无标题"
            pbar.set_postfix_str(f"#{doc_id} {title[:20]}")

            try:
                new_pairs = generate_qa_for_document(doc)
                if new_pairs:
                    all_pairs.extend(new_pairs)
                    completed_ids.add(doc_id)
                    # 每处理完一个文档就保存断点
                    save_checkpoint(completed_ids, all_pairs)
                else:
                    logger.warning("文档 #%d 生成失败（0 对），跳过", doc_id)
            except Exception as exc:
                logger.exception("文档 #%d 处理异常: %s", doc_id, exc)
                # 保存当前进度后继续
                save_checkpoint(completed_ids, all_pairs)

            pbar.update(1)

    # ---- 输出 ----
    write_jsonl(all_pairs, OUTPUT_FILE)

    # ---- 验证 ----
    errors = 0
    for i, entry in enumerate(all_pairs):
        err = validate_qa_pair(entry)
        if err:
            logger.warning("第 %d 条格式错误: %s", i + 1, err)
            errors += 1
    if errors:
        logger.warning("共 %d 条格式错误", errors)
    else:
        logger.info("全部 %d 条记录格式验证通过", len(all_pairs))

    print_stats(all_pairs)

    # 清理断点（全部完成后）
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("已清理断点文件")

    logger.info("========== 完成 ==========")


if __name__ == "__main__":
    main()
