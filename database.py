# 生鲜 AI 品控 — SQLite 数据持久化

import sqlite3
from datetime import datetime
from pathlib import Path

import config as cfg


def get_conn():
    cfg.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_count INTEGER,
            total_defect INTEGER,
            overall_loss_rate TEXT,
            report_text TEXT
        );
        CREATE TABLE IF NOT EXISTS inspection_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            count INTEGER,
            defect_count INTEGER,
            loss_rate TEXT,
            FOREIGN KEY (inspection_id) REFERENCES inspections(id)
        );
    """)
    conn.commit()
    conn.close()


def save_inspection(stats_list: list, stats_summary: dict, report_text: str = ""):
    conn = get_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    cur = conn.execute(
        "INSERT INTO inspections (date, total_count, total_defect, overall_loss_rate, report_text) "
        "VALUES (?, ?, ?, ?, ?)",
        (today,
         stats_summary["total_count"],
         stats_summary["total_defect"],
         stats_summary["overall_loss_rate"],
         report_text),
    )
    insp_id = cur.lastrowid
    for s in stats_list:
        conn.execute(
            "INSERT INTO inspection_items (inspection_id, category, count, defect_count, loss_rate) "
            "VALUES (?, ?, ?, ?, ?)",
            (insp_id, s["category_en"], s["count"], s["defect_count"], s["loss_rate"]),
        )
    conn.commit()
    conn.close()
    return insp_id


def get_recent_history(days: int = 7):
    """获取近 N 天的检测摘要，用于 LLM 上下文。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT date, total_count, total_defect, overall_loss_rate "
        "FROM inspections WHERE date >= date('now', ?) "
        "ORDER BY date DESC", (f"-{days} days",)
    ).fetchall()
    conn.close()
    if not rows:
        return ""
    lines = ["近 7 天历史记录："]
    for r in rows:
        lines.append(f"  {r['date']}: 总数 {r['total_count']}, "
                      f"瑕疵 {r['total_defect']}, 损耗率 {r['overall_loss_rate']}")
    return "\n".join(lines)
