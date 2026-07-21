# 生鲜 AI 品控 — 视觉检测模块
# YOLOv11n 目标检测 + HSV 颜色空间瑕疵分析

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import config as cfg

# ── 全局模型（惰性加载） ──
_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(cfg.MODEL_PATH))
    return _model


# ── 单图检测入口 ──
def detect(image: np.ndarray):
    """返回 (annotated_img, stats_list, stats_summary)

    stats_list: [{"category_cn", "category_en", "count", "defect_count", "loss_rate"}, ...]
    stats_summary: {"total_count", "total_defect", "overall_loss_rate"}
    """
    model = get_model()

    # 1. YOLOv11n 推理
    results = model(
        image,
        conf=cfg.YOLO_CONF_THRESH,
        iou=cfg.YOLO_IOU_THRESH,
        device=cfg.YOLO_DEVICE,
        imgsz=cfg.YOLO_IMG_SIZE,
        verbose=False,
    )
    result = results[0]

    # 2. 过滤目标品类
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return image, [], {"total_count": 0, "total_defect": 0, "overall_loss_rate": 0.0}

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    # 只保留关注的品类
    detections = []
    for cls_id, conf, box in zip(cls_ids, confs, xyxy):
        if cls_id in cfg.COCO_CATEGORIES:
            detections.append({
                "cls_id": cls_id,
                "category_en": cfg.COCO_CATEGORIES[cls_id],
                "conf": float(conf),
                "box": box.tolist(),       # [x1, y1, x2, y2]
            })

    if not detections:
        return image, [], {"total_count": 0, "total_defect": 0, "overall_loss_rate": 0.0}

    # 3. HSV 瑕疵判定
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        # 扩展 5px 边界
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(width - 1, x2 + 5)
        y2 = min(height - 1, y2 + 5)

        roi_hsv = hsv[y1:y2, x1:x2]
        if roi_hsv.size == 0:
            det["is_defect"] = False
            det["defect_ratio"] = 0.0
            continue

        # 获取品类对应的健康 HSV 掩膜
        cat_en = det["category_en"]
        mask_healthy = _build_healthy_mask(roi_hsv, cat_en)
        total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
        healthy_pixels = int(cv2.countNonZero(mask_healthy)) if mask_healthy is not None else 0
        defect_ratio = 1.0 - (healthy_pixels / max(total_pixels, 1))

        det["defect_ratio"] = round(defect_ratio, 4)
        det["is_defect"] = defect_ratio > cfg.DEFECT_THRESHOLD

    # 4. 统计汇总
    cat_stats = {}
    for det in detections:
        cat = det["category_en"]
        if cat not in cat_stats:
            cat_stats[cat] = {"count": 0, "defect_count": 0}
        cat_stats[cat]["count"] += 1
        if det.get("is_defect", False):
            cat_stats[cat]["defect_count"] += 1

    stats_list = []
    total_count = 0
    total_defect = 0
    for cat_en, s in cat_stats.items():
        loss_rate = round(s["defect_count"] / max(s["count"], 1) * 100, 1)
        stats_list.append({
            "category_cn": cfg.CN_CATEGORIES.get(cat_en, cat_en),
            "category_en": cat_en,
            "count": s["count"],
            "defect_count": s["defect_count"],
            "loss_rate": f"{loss_rate}%",
        })
        total_count += s["count"]
        total_defect += s["defect_count"]

    overall_loss = round(total_defect / max(total_count, 1) * 100, 1)
    stats_summary = {
        "total_count": total_count,
        "total_defect": total_defect,
        "overall_loss_rate": f"{overall_loss}%",
    }

    # 5. 绘制标注图
    annotated = _draw_detections(image.copy(), detections)

    return annotated, stats_list, stats_summary


# ── 绘制检测框 ──
def _draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        is_defect = det.get("is_defect", False)
        cat_cn = cfg.CN_CATEGORIES.get(det["category_en"], det["category_en"])
        label = f"{cat_cn} {det.get('conf', 0):.2f}"
        if is_defect:
            label += " [瑕疵]"

        color = (0, 0, 255) if is_defect else (0, 255, 0)  # BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 瑕疵区域着色
        if is_defect:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    return img


# ── 构建健康 HSV 掩膜 ──
def _build_healthy_mask(hsv: np.ndarray, category: str):
    ranges = []
    for key, (lower, upper) in cfg.HEALTHY_HSV.items():
        if key.startswith(category) or key == category:
            ranges.append((np.array(lower, dtype=np.uint8),
                           np.array(upper, dtype=np.uint8)))
    if not ranges:
        return None
    mask = cv2.inRange(hsv, *ranges[0])
    for lower, upper in ranges[1:]:
        mask |= cv2.inRange(hsv, lower, upper)
    return mask


# ── 构建检测结果 JSON（供报表模块使用） ──
def build_report_data(stats_list, stats_summary):
    items = []
    for s in stats_list:
        items.append({
            "品类": s["category_cn"],
            "数量": s["count"],
            "瑕疵数": s["defect_count"],
            "损耗率": s["loss_rate"],
        })
    return {
        "检测数据": {
            "总数量": stats_summary["total_count"],
            "总瑕疵数": stats_summary["total_defect"],
            "整体损耗率": stats_summary["overall_loss_rate"],
            "品类详情": items,
        }
    }
