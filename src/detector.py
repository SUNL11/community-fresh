# 生鲜 AI 品控 — 视觉检测模块
# YOLOv11n 目标检测 + 类别识别与计数

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from configs import config as cfg

_model = None


def get_model():
    global _model
    if _model is None:
        model_path = str(cfg.MODEL_PATH)
        if not cfg.MODEL_PATH.exists():
            msg = "模型文件未找到: {}\n请将 yolo11n.pt 放置在 models/ 目录下".format(cfg.MODEL_PATH)
            raise FileNotFoundError(msg)
        _model = YOLO(model_path)
    return _model


def detect(image: np.ndarray):
    if image is None or image.size == 0:
        raise ValueError("输入图片为空")
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise ValueError("图片尺寸太小 ({}x{})，请上传不小于 32x32 的图片".format(w, h))

    model = get_model()

    try:
        results = model(
            image, conf=cfg.YOLO_CONF_THRESH, iou=cfg.YOLO_IOU_THRESH,
            device=cfg.YOLO_DEVICE, imgsz=cfg.YOLO_IMG_SIZE, verbose=False,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError("GPU 显存不足，请在 config.py 中将 YOLO_DEVICE 设为 cpu")
        raise RuntimeError("模型推理失败: {}".format(e))

    result = results[0]

    # 过滤目标品类
    if result.boxes is None or result.boxes.cls is None:
        return _empty_result(image)

    detections = []
    for cls_id, conf, box in zip(
        result.boxes.cls.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
        result.boxes.xyxy.cpu().numpy(),
    ):
        if cls_id in cfg.COCO_CATEGORIES:
            detections.append({
                "cls_id": cls_id,
                "category_en": cfg.COCO_CATEGORIES[cls_id],
                "conf": float(conf),
                "box": box.tolist(),
            })

    if not detections:
        return _empty_result(image)

    # 按品类统计数量
    cat_counts = {}
    for det in detections:
        cat = det["category_en"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    stats_list = []
    total_count = sum(cat_counts.values())
    for cat_en, count in sorted(cat_counts.items()):
        stats_list.append({
            "category_cn": cfg.CN_CATEGORIES.get(cat_en, cat_en),
            "category_en": cat_en,
            "count": count,
        })

    stats_summary = {"total_count": total_count}

    # 绘制检测框
    annotated = _draw_detections(image.copy(), detections)

    return annotated, stats_list, stats_summary


def _empty_result(image: np.ndarray):
    return image, [], {"total_count": 0}


def _draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cat_en = det["category_en"]
        label = "{} {:.2f}".format(cat_en, det.get("conf", 0))

        color = (0, 200, 255)  # BGR 橙色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def build_report_data(stats_list: list, stats_summary: dict) -> dict:
    items = [{"品类": s["category_cn"], "数量": s["count"]} for s in stats_list]
    return {"检测数据": {"总数量": stats_summary["total_count"], "品类详情": items}}
