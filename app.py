# 生鲜 AI 品控与损耗智能统计工具 — Gradio 主界面

import json

import gradio as gr
import numpy as np

from config import GRADIO_TITLE, GRADIO_PORT, GRADIO_SHARE
from detector import detect, build_report_data
from reporter import generate_report
from database import init_db, save_inspection, get_recent_history as get_db_history

# ── 启动时初始化数据库 ──
init_db()


# ── 检测回调 ──
def run_detection(image: np.ndarray | None):
    if image is None:
        return None, [], "", None

    annotated, stats_list, stats_summary = detect(image)

    if stats_list:
        table_data = [
            [s["category_cn"], s["count"], s["defect_count"], s["loss_rate"]]
            for s in stats_list
        ]
    else:
        table_data = []

    summary = (
        f"**总数量**: {stats_summary['total_count']}   "
        f"**总瑕疵数**: {stats_summary['total_defect']}   "
        f"**整体损耗率**: {stats_summary['overall_loss_rate']}"
    )

    detection_data = build_report_data(stats_list, stats_summary)
    detection_json = json.dumps(detection_data, ensure_ascii=False)

    return annotated, table_data, summary, detection_json


# ── 报表生成回调 ──
def run_report(detection_json: str):
    if not detection_json:
        return "请先上传图片并点击「开始检测」"

    detection_data = json.loads(detection_json)
    items = detection_data.get("检测数据", {}).get("品类详情", [])

    # 构造用于保存的 stats_list / stats_summary
    stats_list = []
    total_count = 0
    total_defect = 0
    for item in items:
        cat_en = item.get("品类", "")
        c = item.get("数量", 0)
        d = item.get("瑕疵数", 0)
        stats_list.append({
            "category_en": cat_en,
            "count": c,
            "defect_count": d,
            "loss_rate": item.get("损耗率", "0%"),
        })
        total_count += c
        total_defect += d

    stats_summary = {
        "total_count": total_count,
        "total_defect": total_defect,
        "overall_loss_rate": f"{round(total_defect / max(total_count,1) * 100, 1)}%",
    }

    history = get_db_history(7)
    report = generate_report(detection_data, history)

    # 持久化
    if stats_list:
        save_inspection(stats_list, stats_summary, report)

    return report


# ── 构建 Gradio 界面 ──
with gr.Blocks(title=GRADIO_TITLE, theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"# 🥬 {GRADIO_TITLE}")
    gr.Markdown("上传生鲜图片或使用摄像头拍照，自动识别品类、统计数量、检测瑕疵，一键生成损耗报表。")

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_input = gr.Image(
                label="📤 上传图片 / 拍照",
                sources=["upload", "webcam"],
                type="numpy",
                height=360,
            )
            detect_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

        with gr.Column(scale=2):
            annotated_output = gr.Image(
                label="📊 检测结果",
                type="numpy",
                height=360,
            )

    stats_table = gr.Dataframe(
        label="📋 数据统计",
        headers=["品类", "数量", "瑕疵数", "损耗率"],
        col_count=(4, "fixed"),
        row_count=5,
        interactive=False,
    )
    summary_md = gr.Markdown("")

    with gr.Row():
        report_btn = gr.Button("📄 生成损耗报表", variant="secondary", size="lg")
        report_output = gr.Textbox(
            label="损耗分析报表",
            lines=10,
            max_lines=18,
            show_copy_button=True,
        )

    # ── 状态保持 ──
    detection_state = gr.State("")

    # ── 事件绑定 ──
    detect_btn.click(
        fn=run_detection,
        inputs=[image_input],
        outputs=[annotated_output, stats_table, summary_md, detection_state],
    )

    report_btn.click(
        fn=run_report,
        inputs=[detection_state],
        outputs=[report_output],
    )


# ── 启动 ──
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
    )
