# 生鲜 AI 品控 — 智能损耗报表生成（阿里云百炼）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import httpx

from configs import config as cfg

REPORT_SYSTEM_PROMPT = "你是生鲜门店损耗分析专家「鲜小智」，擅长从数据中提炼 actionable 的运营建议。"
REPORT_USER_PROMPT_TPL = (
    "请根据以下今日检测数据，生成一份简洁的日度损耗分析报表。\n"
    "检测数据：{json_data}\n\n"
    "报表结构：\n"
    "=== 今日损耗概览 ===\n"
    "一句话概括整体情况，包含总数量、瑕疵总数、整体损耗率。如果有历史数据可以进行对比分析。\n\n"
    "=== 重点损耗品类分析 ===\n"
    "指出损耗率最高的品类，结合品类特性（如番茄皮薄易损、香蕉怕压等）推测可能原因。\n\n"
    "=== 管控优化建议 ===\n"
    "输出 2-3 条贴合社区生鲜小店场景的实操建议，语言接地气、可直接执行。\n\n"
    "要求：语言简洁专业，适合门店经营者阅读，总字数 300 字以内。"
)


def generate_report(detection_data: dict, history_summary: str = "") -> str:
    api_key = cfg.DASHSCOPE_API_KEY
    if not api_key:
        return "[错误] 未配置阿里云百炼 API Key，请在 .env 中设置 DASHSCOPE_API_KEY"

    json_str = json.dumps(detection_data.get("检测数据", detection_data),
                          ensure_ascii=False, indent=2)
    user_msg = REPORT_USER_PROMPT_TPL.format(json_data=json_str)
    if history_summary:
        user_msg += f"\n\n历史参考数据：\n{history_summary}"

    messages = [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    payload = {
        "model": cfg.DASHSCOPE_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=cfg.LLM_TIMEOUT) as client:
            resp = client.post(
                f"{cfg.DASHSCOPE_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
        if resp.status_code != 200:
            return f"[错误] 百炼 API 返回 {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "[错误] API 返回为空"
        return choices[0].get("message", {}).get("content", "").strip()
    except httpx.TimeoutException:
        return "[错误] 请求百炼 API 超时，请检查网络后重试"
    except Exception as e:
        return f"[错误] 生成报表时发生异常: {e}"


def format_summary(stats_summary: dict) -> str:
    return (
        f"总数量: {stats_summary['total_count']}, "
        f"瑕疵数: {stats_summary['total_defect']}, "
        f"损耗率: {stats_summary['overall_loss_rate']}"
    )
