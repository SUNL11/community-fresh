# 生鲜 AI 品控与损耗智能统计工具 — Gradio 主界面
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Optional

import gradio as gr
import numpy as np

from configs.config import GRADIO_TITLE, GRADIO_PORT, GRADIO_SHARE
from src.detector import detect, build_report_data
from src.reporter import generate_report
from src.database import init_db, save_inspection, get_recent_history as get_db_history

init_db()


def run_detection(upload_img: Optional[np.ndarray], webcam_img: Optional[np.ndarray]):
    image = upload_img if upload_img is not None else webcam_img
    if image is None:
        gr.Error("请先上传图片或使用摄像头拍照")
        return None, [], "请先上传图片或拍照", ""
    try:
        annotated, stats_list, stats_summary = detect(image)
    except FileNotFoundError:
        gr.Error("模型文件未找到，请确认 models/yolo11n.pt 存在")
        return image, [], "模型文件缺失", ""
    except Exception as e:
        gr.Error("检测失败: {}".format(e))
        return image, [], "检测失败", ""
    if not stats_list:
        gr.Warning("未识别到目标，请换一张图片重试")
        return image, [], "未识别到目标", ""
    table_data = [[s["category_cn"], s["count"]] for s in stats_list]
    summary = "**总数量**: {}".format(stats_summary["total_count"])
    detection_data = build_report_data(stats_list, stats_summary)
    detection_json = json.dumps(detection_data, ensure_ascii=False)
    return annotated, table_data, summary, detection_json


def run_report(detection_json: str):
    if not detection_json:
        return "请先检测图片"
    try:
        detection_data = json.loads(detection_json)
    except json.JSONDecodeError:
        return "检测数据异常，请重新检测"
    items = detection_data.get("检测数据", {}).get("品类详情", [])
    stats_list = []
    total_count = sum(item.get("数量", 0) for item in items)
    for item in items:
        stats_list.append({"category_en": item.get("品类", ""), "count": item.get("数量", 0)})
    stats_summary = {"total_count": total_count, "total_defect": 0, "overall_loss_rate": "0%"}
    history = get_db_history(7)
    report = generate_report(detection_data, history)
    if stats_list:
        save_inspection(stats_list, stats_summary, report)
    gr.Warning("报表已自动保存至本地数据库")
    return report


with gr.Blocks(title=GRADIO_TITLE) as demo:
    gr.Markdown("# 生鲜 AI 品控与损耗智能统计工具")
    gr.Markdown("上传生鲜图片，自动识别品类并统计数量。")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            upload_image = gr.Image(source="upload", type="numpy", height=200, label="上传图片")
            gr.Markdown("")  # spacer
            webcam_image = gr.Image(source="webcam", type="numpy", height=200, label="摄像头拍照")
        with gr.Column(scale=1):
            annotated_output = gr.Image(label="检测结果", type="numpy", height=360)
    detect_btn = gr.Button("开始检测", variant="primary")
    stats_table = gr.Dataframe(
        label="数据统计",
        headers=["品类", "数量"],
        col_count=(2, "fixed"),
        row_count=5,
        interactive=False,
    )
    summary_md = gr.Markdown("")
    report_btn = gr.Button("生成损耗报表", variant="secondary")
    report_output = gr.Textbox(label="损耗分析报表", lines=10, max_lines=18)
    detection_state = gr.State("")

    detect_btn.click(
        fn=run_detection,
        inputs=[upload_image, webcam_image],
        outputs=[annotated_output, stats_table, summary_md, detection_state],
    )
    report_btn.click(
        fn=run_report,
        inputs=[detection_state],
        outputs=[report_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=GRADIO_PORT, share=GRADIO_SHARE)
