# 生鲜 AI 品控 — 全局配置

import os
from pathlib import Path

# ── 项目路径 ──
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
DB_PATH = BASE_DIR / "data" / "inspections.db"
SAMPLE_DIR = BASE_DIR / "static" / "samples"

# ── YOLO 检测参数 ──
YOLO_CONF_THRESH = 0.45           # 置信度阈值
YOLO_IOU_THRESH = 0.50            # NMS 去重 IoU 阈值
YOLO_DEVICE = "cuda:0"            # GPU 设备；设为 "cpu" 可兜底
YOLO_IMG_SIZE = 640               # 输入尺寸

# ── COCO 品类映射 ──
# 只保留我们关注的 4 种生鲜
COCO_CATEGORIES = {
    47: "apple",
    49: "orange",
    52: "banana",
    60: "tomato",
}
CN_CATEGORIES = {
    "apple": "苹果",
    "orange": "橙子",
    "banana": "香蕉",
    "tomato": "番茄",
}

# ── HSV 瑕疵判定 ──
# 每个品类健康果皮的颜色范围（H, S, V 下限／上限）
DEFECT_THRESHOLD = 0.06           # 非健康像素占比超过此值即判为瑕疵
DEFECT_ADAPTIVE_WINDOW = 30       # 自适应阈值滑动窗口大小

HEALTHY_HSV = {
    "apple":  ((0, 10, 50), (10, 255, 255)),      # 红色苹果
    "apple2": ((160, 50, 50), (180, 255, 255)),    # 红色苹果（H 环）
    "orange": ((5, 40, 50), (25, 255, 255)),
    "banana": ((25, 30, 30), (40, 200, 255)),
    "tomato": ((0, 40, 40), (10, 255, 255)),
    "tomato2":((160, 40, 40), (180, 255, 255)),
}

# ── 阿里云百炼 ──
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen-plus"       # 可选 qwen-turbo / qwen-max
LLM_TIMEOUT = 30                     # 请求超时（秒）

# ── Gradio ──
GRADIO_TITLE = "生鲜 AI 品控与损耗智能统计工具"
GRADIO_PORT = 7860
GRADIO_SHARE = False
