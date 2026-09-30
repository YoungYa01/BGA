#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGA 焊点内部孔洞/气泡精度评估工具 (BGA_Void_Detection 本地精简适配版)

功能：
1. 默认读取 data/new-img (图像) 和 data/new-coord (LabelMe 标注 JSON)；
2. 直接调用本地 bga_void_seg.py 进行焊球与气泡高召回推理；
3. 输出实例级 TP / FP / FN，Precision / Recall / F1 / ACC；
4. 输出像素级 Mask IoU / Dice；
5. 输出气泡率误差 (void_rate MAE) 与 NG 判定准确率；
6. 生成可视化图片 (TP绿色、FP红色、FN橙色) 及 CSV / JSON 报表。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 确保能直接导入本地模块
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bga_void_seg

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


@dataclass
class Circle:
    x: float
    y: float
    r: float
    label: str = ""
    parent_index: int = -1
    confidence: float = 0.0
    void_rate: float = 0.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def f1_score(p: float, r: float) -> float:
    return safe_div(2.0 * p * r, p + r)


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    buf.tofile(str(path))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_labelme_circles(data: Dict[str, Any], label: str) -> List[Circle]:
    out: List[Circle] = []
    for shape in data.get("shapes", []):
        if str(shape.get("label", "")).strip() != label:
            continue
        pts = shape.get("points") or []
        if len(pts) < 2:
            continue
        try:
            x0, y0 = float(pts[0][0]), float(pts[0][1])
            x1, y1 = float(pts[1][0]), float(pts[1][1])
        except Exception:
            continue
        r = math.hypot(x1 - x0, y1 - y0)
        if r <= 0:
            continue
        out.append(Circle(x=x0, y=y0, r=r, label=label))
    return out


def deduplicate_circles(circles: Sequence[Circle], dist_thresh: float = 3.0) -> List[Circle]:
    """消除标注数据中偶发的人工重复标注圈 (距离小于 dist_thresh 像素视为重复)"""
    clean: List[Circle] = []
    for c in circles:
        if not any(math.hypot(c.x - u.x, c.y - u.y) < dist_thresh for u in clean):
            clean.append(c)
    return clean


def scale_circles(circles: Sequence[Circle], sx: float, sy: float) -> List[Circle]:
    sr = math.sqrt(max(1e-12, sx * sy))
    return [
        Circle(
            x=c.x * sx, y=c.y * sy, r=c.r * sr, label=c.label,
            parent_index=c.parent_index, confidence=c.confidence, void_rate=c.void_rate
        )
        for c in circles
    ]


def assign_holes_to_gt_solders(
    holes: Sequence[Circle],
    solders: Sequence[Circle],
    contain_ratio: float = 1.05,
) -> Tuple[List[Circle], List[Circle]]:
    inside: List[Circle] = []
    outside: List[Circle] = []
    for h in holes:
        best = None
        best_score = float("inf")
        for i, s in enumerate(solders):
            d = math.hypot(h.x - s.x, h.y - s.y)
            lim = max(1.0, s.r * contain_ratio)
            if d <= lim:
                score = d / max(1e-6, s.r)
                if score < best_score:
                    best_score = score
                    best = i
        c = Circle(**asdict(h))
        if best is None:
            outside.append(c)
        else:
            c.parent_index = int(best)
            inside.append(c)
    return inside, outside


def circle_intersection_area(a: Circle, b: Circle) -> float:
    r1, r2 = float(a.r), float(b.r)
    d = math.hypot(a.x - b.x, a.y - b.y)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    if d <= 1e-12:
        return math.pi * min(r1, r2) ** 2

    x1 = clamp((d*d + r1*r1 - r2*r2) / (2*d*r1), -1.0, 1.0)
    x2 = clamp((d*d + r2*r2 - r1*r1) / (2*d*r2), -1.0, 1.0)
    part1 = r1*r1 * math.acos(x1)
    part2 = r2*r2 * math.acos(x2)
    part3 = 0.5 * math.sqrt(max(0.0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
    return part1 + part2 - part3


def circle_iou(a: Circle, b: Circle) -> float:
    inter = circle_intersection_area(a, b)
    union = math.pi * a.r * a.r + math.pi * b.r * b.r - inter
    return safe_div(inter, union)


def eligible_center_match(
    gt: Circle,
    pred: Circle,
    radius_ratio: float,
    min_px: float,
    max_px: float,
) -> Tuple[bool, float, float]:
    # 1. 尺寸一致性约束：半径差异不可超过 2.8 倍
    # 杜绝大假圆 (如 r=38) 恶意捕获并冒领微小真实孔洞 (如 r=6.8) 导致染绿的严重缺陷
    r_min = min(gt.r, pred.r)
    r_max = max(gt.r, pred.r)
    if r_max <= 0 or (r_min / r_max) < 0.35:
        return False, float("inf"), 0.0

    # 2. 距离容差严格以 GT 真实尺寸为基准，避免预测大圆撑大搜索容差
    d = math.hypot(gt.x - pred.x, gt.y - pred.y)
    thr = clamp(radius_ratio * gt.r + 2.0, min_px, max_px)
    return d <= thr, d, thr


def greedy_match_center(
    gt: Sequence[Circle],
    pred: Sequence[Circle],
    radius_ratio: float,
    min_px: float,
    max_px: float,
):
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            ok, d, thr = eligible_center_match(g, p, radius_ratio, min_px, max_px)
            if ok:
                iou = circle_iou(g, p)
                candidates.append((d / max(thr, 1e-9), -iou, d, gi, pi, thr, iou))
    candidates.sort()

    used_g, used_p = set(), set()
    matches = []
    for _, _, d, gi, pi, thr, iou in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append({
            "gt_index": gi,
            "pred_index": pi,
            "center_distance": float(d),
            "match_threshold": float(thr),
            "circle_iou": float(iou),
        })
    return matches, sorted(set(range(len(gt))) - used_g), sorted(set(range(len(pred))) - used_p)


def greedy_match_iou(gt: Sequence[Circle], pred: Sequence[Circle], min_iou: float):
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            iou = circle_iou(g, p)
            if iou >= min_iou:
                d = math.hypot(g.x - p.x, g.y - p.y)
                candidates.append((-iou, d, gi, pi))
    candidates.sort()

    used_g, used_p = set(), set()
    matches = []
    for neg_iou, d, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append({
            "gt_index": gi,
            "pred_index": pi,
            "center_distance": float(d),
            "match_threshold": None,
            "circle_iou": float(-neg_iou),
        })
    return matches, sorted(set(range(len(gt))) - used_g), sorted(set(range(len(pred))) - used_p)


def circles_to_mask(shape_hw: Tuple[int, int], circles: Sequence[Circle]) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    for c in circles:
        cx = int(round(c.x))
        cy = int(round(c.y))
        r = max(1, int(round(c.r)))
        cv2.circle(m, (cx, cy), r, 255, thickness=-1)
    return m


def parse_predictions(raw_results: Sequence[Dict[str, Any]]) -> Tuple[List[Circle], List[Circle]]:
    pred_solders: List[Circle] = []
    pred_holes: List[Circle] = []

    for si, result in enumerate(raw_results):
        sc = result.get("solder_circle") or {}
        try:
            center = sc.get("center")
            sx, sy = float(center[0]), float(center[1])
            sr = float(sc.get("radius", 0))
        except Exception:
            continue
        if sr <= 0:
            continue

        conf = float(result.get("confidence", 0.0) or 0.0)
        vrate = float(result.get("void_rate", 0.0) or 0.0)
        pred_solders.append(Circle(sx, sy, sr, label="pred_solder", parent_index=si, confidence=conf, void_rate=vrate))

        for vc in result.get("void_circle") or []:
            try:
                center = vc.get("center")
                vx, vy = float(center[0]), float(center[1])
                vr = float(vc.get("radius", 0))
            except Exception:
                continue
            if vr <= 0:
                continue
            pred_holes.append(Circle(vx, vy, vr, label="pred_hole", parent_index=si, confidence=conf, void_rate=vrate))

    return pred_solders, pred_holes


def match_solders_for_rate(
    gt_solders: Sequence[Circle],
    pred_solders: Sequence[Circle],
    ratio: float = 0.75,
    min_px: float = 6.0,
    max_px: float = 35.0,
):
    return greedy_match_center(gt_solders, pred_solders, ratio, min_px, max_px)


def gt_void_rate_for_solder(
    image_shape_hw: Tuple[int, int],
    solder: Circle,
    gt_holes: Sequence[Circle],
) -> float:
    h, w = image_shape_hw
    x1 = max(0, int(math.floor(solder.x - solder.r - 2)))
    y1 = max(0, int(math.floor(solder.y - solder.r - 2)))
    x2 = min(w, int(math.ceil(solder.x + solder.r + 3)))
    y2 = min(h, int(math.ceil(solder.y + solder.r + 3)))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    hh, ww = y2 - y1, x2 - x1
    sm = np.zeros((hh, ww), dtype=np.uint8)
    hm = np.zeros((hh, ww), dtype=np.uint8)
    cv2.circle(sm, (int(round(solder.x - x1)), int(round(solder.y - y1))), max(1, int(round(solder.r))), 255, -1)

    for hole in gt_holes:
        if hole.parent_index < 0:
            continue
        cv2.circle(hm, (int(round(hole.x - x1)), int(round(hole.y - y1))), max(1, int(round(hole.r))), 255, -1)

    hm = cv2.bitwise_and(hm, sm)
    sa = int((sm > 0).sum())
    va = int((hm > 0).sum())
    return safe_div(va, sa)


def draw_eval(
    image: np.ndarray,
    gt_eval: Sequence[Circle],
    pred_holes: Sequence[Circle],
    matches: Sequence[Dict[str, Any]],
    unmatched_gt: Sequence[int],
    unmatched_pred: Sequence[int],
    metrics: Dict[str, Any],
    gt_solders: Optional[Sequence[Circle]] = None,
    pred_solders: Optional[Sequence[Circle]] = None,
    unmatched_gt_solders: Optional[Sequence[int]] = None,
    unmatched_pred_solders: Optional[Sequence[int]] = None,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    绘制单张评估对比标注图：
    1. 焊球三分类标注：
       - 原数据集中有且被检测到的焊球 (Matched Solder)：淡青色细圈 (220, 220, 70)，线宽 1px；
       - 原数据集中有但未被检测到的漏检焊球 (Missed Solder / unmatched_gt_solders)：洋红色粗圈 (255, 0, 255)，线宽 2px；
       - 原数据集中没有但被模型检测到的多检/虚报焊球 (Extra Pred Solder / unmatched_pred_solders)：天蓝色粗圈 (255, 140, 0)，线宽 2px；
    2. 孔洞三分类标注：
       - 正确匹配孔洞 (TP)：绿色纯圆圈 (0, 255, 0)；
       - 误检孔洞 (FP)：红色纯圆圈 (0, 0, 255)；
       - 漏检孔洞 (FN)：橙黄色纯圆圈 (0, 165, 255)；
    3. 采用精细抗锯齿细线，去除一切内部十字，保持画面干净通透。
    """
    h, w = image.shape[:2]
    vis = image.copy()

    unmatched_pred_set = set(unmatched_pred_solders or [])

    # 1. 绘制检测到的焊球
    if pred_solders is not None and len(pred_solders) > 0:
        for pi, s in enumerate(pred_solders):
            if pi in unmatched_pred_set:
                # 原数据集中没有，但模型检测到的焊球 (多检/虚报焊球，用天蓝色 Sky Blue 醒目标注，线宽 2px)
                cv2.circle(vis, (int(round(s.x)), int(round(s.y))), max(1, int(round(s.r))), (255, 140, 0), 2, cv2.LINE_AA)
            else:
                # 原数据集中有且被检测到的焊球 (正常匹配焊球，用淡青色细圈 1px)
                cv2.circle(vis, (int(round(s.x)), int(round(s.y))), max(1, int(round(s.r))), (220, 220, 70), 1, cv2.LINE_AA)
    elif gt_solders:
        for s in gt_solders:
            cv2.circle(vis, (int(round(s.x)), int(round(s.y))), max(1, int(round(s.r))), (220, 220, 70), 1, cv2.LINE_AA)

    # 2. 醒目标注漏检的焊球 (原数据集存在但 YOLO 检测器漏检，用洋红色 Magenta 圈出，线宽 2px)
    if gt_solders is not None and unmatched_gt_solders is not None and len(unmatched_gt_solders) > 0:
        for gi in unmatched_gt_solders:
            s = gt_solders[gi]
            cv2.circle(vis, (int(round(s.x)), int(round(s.y))), max(1, int(round(s.r))), (255, 0, 255), 2, cv2.LINE_AA)

    # 3. TP：正确匹配孔洞 (绿色纯圆圈)
    for m in matches:
        p = pred_holes[m["pred_index"]]
        px, py = int(round(p.x)), int(round(p.y))
        draw_r = max(2, int(round(p.r)))
        cv2.circle(vis, (px, py), draw_r, (0, 255, 0), line_thickness, cv2.LINE_AA)

    # 4. FP：预测误检孔洞 (红色纯圆圈)
    for pi in unmatched_pred:
        p = pred_holes[pi]
        px, py = int(round(p.x)), int(round(p.y))
        draw_r = max(2, int(round(p.r)))
        cv2.circle(vis, (px, py), draw_r, (0, 0, 255), line_thickness, cv2.LINE_AA)

    # 5. FN：真实漏检孔洞 (橙黄色纯圆圈)
    for gi in unmatched_gt:
        g = gt_eval[gi]
        gx, gy = int(round(g.x)), int(round(g.y))
        draw_r = max(2, int(round(g.r)))
        cv2.circle(vis, (gx, gy), draw_r, (0, 165, 255), line_thickness, cv2.LINE_AA)

    # 6. 左上角紧凑状态信息框 + 彩色图例
    missed_solder_cnt = len(unmatched_gt_solders) if unmatched_gt_solders else 0
    extra_solder_cnt = len(unmatched_pred_solders) if unmatched_pred_solders else 0
    rec_area_str = f"{metrics.get('recall_area', metrics['recall']) * 100:.1f}%"
    rec_cnt_str = f"{metrics.get('recall_count', 0.0) * 100:.1f}%"
    info_row1 = (
        f"[{metrics['image']}]  "
        f"GT: {metrics['gt_count']} | Pred: {metrics['pred_count']} | "
        f"TP: {metrics['tp']} FP: {metrics['fp']} FN: {metrics['fn']} | "
        f"Recall(Area): {rec_area_str} (Num: {rec_cnt_str})  P: {metrics['precision']*100:.1f}%  "
        f"F1: {metrics['f1']*100:.1f}% | Solder Missed: {missed_solder_cnt}  Extra: {extra_solder_cnt}"
    )

    box_w = min(w - 10, 1020)
    box_h = 44
    cv2.rectangle(vis, (6, 6), (6 + box_w, 6 + box_h), (20, 22, 26), -1)
    cv2.rectangle(vis, (6, 6), (6 + box_w, 6 + box_h), (60, 65, 70), 1)

    # 第一行：指标文本
    cv2.putText(vis, info_row1, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)

    # 第二行：图例 (Legend)
    lg_y = 42
    # TP (绿色)
    cv2.circle(vis, (18, lg_y - 4), 4, (0, 255, 0), -1)
    cv2.putText(vis, "TP (Matched)", (28, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
    # FP (红色)
    cv2.circle(vis, (125, lg_y - 4), 4, (0, 0, 255), -1)
    cv2.putText(vis, "FP (False)", (135, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
    # FN (橙色)
    cv2.circle(vis, (215, lg_y - 4), 4, (0, 165, 255), -1)
    cv2.putText(vis, "FN (Missed)", (225, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)
    # Solder Matched (淡青色细圈 1px)
    cv2.circle(vis, (315, lg_y - 4), 4, (220, 220, 70), 1, cv2.LINE_AA)
    cv2.putText(vis, "Solder Matched", (325, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 70), 1, cv2.LINE_AA)
    # Solder Missed (洋红色圈 2px)
    cv2.circle(vis, (435, lg_y - 4), 4, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Solder Missed", (445, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 255), 1, cv2.LINE_AA)
    # Solder Extra (天蓝色圈 2px)
    cv2.circle(vis, (550, lg_y - 4), 4, (255, 140, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Solder Extra", (560, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 140, 0), 1, cv2.LINE_AA)

    return vis


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fields})
    except PermissionError:
        alt_path = path.with_name(f"{path.stem}_{int(time.time())}.csv")
        print(f"\n[提示] {path.name} 被其他软件占用锁定，已自动另存为: {alt_path.name}")
        with alt_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fields})


def discover_image_coord_pairs(
    img_dir: Path,
    coord_dir: Path,
) -> Tuple[List[Tuple[Path, Path, Dict[str, Any]]], List[Dict[str, str]]]:
    pairs = []
    skipped = []

    if not img_dir.is_dir():
        return pairs, [{"json": "", "reason": f"图片目录不存在: {img_dir}"}]
    if not coord_dir.is_dir():
        return pairs, [{"json": "", "reason": f"标注目录不存在: {coord_dir}"}]

    json_files = {p.stem: p for p in coord_dir.glob("*.json")}
    for ip in sorted(img_dir.iterdir()):
        if not ip.is_file() or ip.suffix.lower() not in IMAGE_EXTS:
            continue
        jp = json_files.get(ip.stem)
        if jp is None:
            skipped.append({"json": "", "reason": f"图片 {ip.name} 未找到对应 JSON"})
            continue
        try:
            data = load_json(jp)
        except Exception as e:
            skipped.append({"json": str(jp), "reason": f"JSON读取失败: {e}"})
            continue

        if "shapes" not in data:
            continue
        pairs.append((jp.resolve(), ip.resolve(), data))

    return pairs, skipped


def main():
    parser = argparse.ArgumentParser(description="BGA 焊点内部孔洞/气泡高召回精度评估工具")
    parser.add_argument("--img-dir", default=str(HERE / "data" / "new-img"), help="图片目录路径 (默认: data/new-img)")
    parser.add_argument("--coord-dir", default=str(HERE / "data" / "new-coord"), help="标注 JSON 目录路径 (默认: data/new-coord)")
    parser.add_argument("--weights", default=str(HERE / "best.pt"), help="模型权重文件路径 (默认: best.pt)")
    parser.add_argument("--output", default=str(HERE / "output" / "void_eval_results"), help="评估结果输出目录")
    parser.add_argument("--device", default="cpu", help="推理设备 (cpu, 0 等)")
    parser.add_argument("--conf", type=float, default=0.25, help="焊点置信度阈值 (默认: 0.25)")
    parser.add_argument("--hole-label", default="hole", help="孔洞标签名称 (默认: hole)")
    parser.add_argument("--solder-label", default="solder ball", help="焊球标签名称 (默认: solder ball)")
    parser.add_argument("--gt-hole-scope", choices=["inside_solder_ball", "all"], default="inside_solder_ball", help="GT 评估范围")
    parser.add_argument("--contain-ratio", type=float, default=1.05, help="焊球包含孔洞的判定比例")
    parser.add_argument("--match-mode", choices=["center", "iou"], default="center", help="匹配模式 (center/iou)")
    parser.add_argument("--match-radius-ratio", type=float, default=0.75, help="圆心匹配自适应半径倍数 (默认: 0.75)")
    parser.add_argument("--match-min-px", type=float, default=3.0, help="圆心匹配最小容差像素 (默认: 3.0)")
    parser.add_argument("--match-max-px", type=float, default=8.0, help="圆心匹配最大容差像素 (默认: 8.0)")
    parser.add_argument("--match-iou", type=float, default=0.10, help="IoU 匹配阈值")
    parser.add_argument("--json-size-policy", choices=["scale", "error", "keep"], default="scale", help="尺寸不一致时的缩放策略")
    parser.add_argument("--ng-threshold", type=float, default=0.25, help="NG 判定气泡率阈值 (默认 0.25)")
    parser.add_argument("--line-thickness", type=int, default=1, help="标注轮廓线条粗细 (默认: 1)")
    parser.add_argument("--save-collaborator-debug", action="store_true", help="是否保存算法内部调试图")
    parser.add_argument("--dry-run", action="store_true", help="仅预检数据匹配，不运行模型")
    args = parser.parse_args()

    img_dir = Path(args.img_dir).resolve()
    coord_dir = Path(args.coord_dir).resolve()
    output = Path(args.output).resolve()
    weights = Path(args.weights).resolve()

    if not weights.is_file() and not args.dry_run:
        raise SystemExit(f"❌ 权重文件不存在: {weights}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "visualizations").mkdir(exist_ok=True)
    (output / "per_image").mkdir(exist_ok=True)

    pairs, skipped = discover_image_coord_pairs(img_dir, coord_dir)
    if not pairs:
        raise SystemExit(f"❌ 未在 {img_dir} 和 {coord_dir} 中找到有效图片-JSON 配对！")

    print("=" * 60)
    print("      BGA 焊点气泡高召回分割精度评估 (BGA_Void_Detection)")
    print("=" * 60)
    print(f"图像目录   : {img_dir}")
    print(f"标注目录   : {coord_dir}")
    print(f"输出目录   : {output}")
    print(f"模型权重   : {weights}")
    print(f"设备       : {args.device}")
    print(f"置信度     : {args.conf}")
    print(f"有效测试样本: {len(pairs)} 对")
    print("-" * 60)

    if args.dry_run:
        print("Dry-run 检查完成。")
        return 0

    rows = []
    totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    mask_inters = 0
    mask_unions = 0
    mask_gt_sum = 0
    mask_pred_sum = 0
    total_gt_circle_area = 0.0
    total_matched_gt_circle_area = 0.0
    matched_center_errors = []
    matched_circle_ious = []
    rate_abs_errors = []
    ng_correct = 0
    ng_total = 0
    total_matched_solders_for_rate = 0

    t_start = time.perf_counter()

    for idx, (jp, ip, jd) in enumerate(pairs, 1):
        img = imread_unicode(ip)
        if img is None:
            print(f"[{idx:2d}/{len(pairs):2d}] ❌ 读取图片失败: {ip.name}")
            continue

        h, w = img.shape[:2]
        holes_all = deduplicate_circles(parse_labelme_circles(jd, args.hole_label))
        gt_solders = deduplicate_circles(parse_labelme_circles(jd, args.solder_label))

        jw = int(jd.get("imageWidth") or 0)
        jh = int(jd.get("imageHeight") or 0)
        size_note = ""
        if jw > 0 and jh > 0 and (jw != w or jh != h):
            if args.json_size_policy == "scale":
                sx, sy = w / float(jw), h / float(jh)
                holes_all = scale_circles(holes_all, sx, sy)
                gt_solders = scale_circles(gt_solders, sx, sy)
                size_note = f"scaled_{jw}x{jh}_to_{w}x{h}"

        inside_holes, outside_holes = assign_holes_to_gt_solders(
            holes_all, gt_solders, args.contain_ratio
        )
        gt_eval = inside_holes if args.gt_hole_scope == "inside_solder_ball" else holes_all

        # 调用本地高召回自适应算法
        t0 = time.perf_counter()
        raw_results = bga_void_seg.predict_and_generate_mask(
            str(weights),
            str(ip),
            float(args.conf),
            device=str(args.device),
            save_debug_image=bool(args.save_collaborator_debug),
            debug_output_dir=str(output / "collaborator_debug"),
            ng_threshold=float(args.ng_threshold * 100.0),
        )
        infer_s = time.perf_counter() - t0

        pred_solders, pred_holes = parse_predictions(raw_results)

        # 匹配孔洞
        if args.match_mode == "center":
            matches, unmatched_gt, unmatched_pred = greedy_match_center(
                gt_eval, pred_holes,
                args.match_radius_ratio, args.match_min_px, args.match_max_px,
            )
        else:
            matches, unmatched_gt, unmatched_pred = greedy_match_iou(
                gt_eval, pred_holes, args.match_iou
            )

        # 1. 实例数量级匹配 (Count-based)
        tp_count = len(matches)
        fp_count = len(unmatched_pred)
        fn_count = len(unmatched_gt)
        precision_count = safe_div(tp_count, tp_count + fp_count)
        recall_count = safe_div(tp_count, tp_count + fn_count)
        f1_count = f1_score(precision_count, recall_count)
        accuracy_count = safe_div(tp_count, tp_count + fp_count + fn_count)

        for m in matches:
            matched_center_errors.append(float(m["center_distance"]))
            matched_circle_ious.append(float(m["circle_iou"]))

        # 2. 掩膜像素面积级指标 (Area-based)
        gt_mask = circles_to_mask((h, w), gt_eval)
        pred_mask = circles_to_mask((h, w), pred_holes)
        g = gt_mask > 0
        p = pred_mask > 0
        inter = int(np.logical_and(g, p).sum())
        union = int(np.logical_or(g, p).sum())
        gsum = int(g.sum())
        psum = int(p.sum())
        miou = safe_div(inter, union)
        mdice = safe_div(2 * inter, gsum + psum)

        mask_inters += inter
        mask_unions += union
        mask_gt_sum += gsum
        mask_pred_sum += psum

        # 🌟 面积召回率 (inter/gsum)、面积精确率 (inter/psum)、面积 F1 (mdice)
        recall_area = safe_div(inter, gsum)
        precision_area = safe_div(inter, psum)
        f1_area = mdice
        accuracy_area = miou

        # 3. 气泡圆面积加权召回率 (已检出真实气泡圆面积占比)
        all_gt_circle_area = sum(math.pi * (c.r ** 2) for c in gt_eval)
        matched_gt_indices = {m["gt_index"] for m in matches}
        matched_gt_circle_area = sum(math.pi * (gt_eval[i].r ** 2) for i in matched_gt_indices)
        recall_circle_area = safe_div(matched_gt_circle_area, all_gt_circle_area)

        total_gt_circle_area += all_gt_circle_area
        total_matched_gt_circle_area += matched_gt_circle_area

        # 核心指标采用面积计算
        recall = recall_area
        precision = precision_area
        f1 = f1_area
        accuracy = accuracy_area

        # void_rate 比较
        solder_matches, unmatched_gt_solders, unmatched_pred_solders = match_solders_for_rate(gt_solders, pred_solders)
        image_rate_errors = []
        image_ng_correct = 0
        for sm in solder_matches:
            gi, pi = sm["gt_index"], sm["pred_index"]
            gt_s = gt_solders[gi]
            assigned = [hh for hh in inside_holes if hh.parent_index == gi]
            gt_rate = gt_void_rate_for_solder((h, w), gt_s, assigned)
            pred_rate = float(pred_solders[pi].void_rate)
            ae = abs(gt_rate - pred_rate)
            rate_abs_errors.append(ae)
            image_rate_errors.append(ae)
            total_matched_solders_for_rate += 1

            gt_ng = gt_rate > args.ng_threshold
            pred_ng = pred_rate > args.ng_threshold
            ok = int(gt_ng == pred_ng)
            ng_correct += ok
            image_ng_correct += ok
            ng_total += 1

        metrics = {
            "image": ip.name,
            "json": jp.name,
            "width": w,
            "height": h,
            "size_note": size_note,
            "gt_solder_count": len(gt_solders),
            "gt_count": len(gt_eval),
            "pred_solder_count": len(pred_solders),
            "pred_count": len(pred_holes),
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "precision": precision_area,
            "recall": recall_area,        # 🌟 召回率采用面积计算
            "f1": f1_area,
            "accuracy": accuracy_area,
            "recall_area": recall_area,
            "recall_circle_area": recall_circle_area,
            "recall_count": recall_count,
            "precision_area": precision_area,
            "precision_count": precision_count,
            "f1_area": f1_area,
            "f1_count": f1_count,
            "accuracy_area": accuracy_area,
            "accuracy_count": accuracy_count,
            "mask_iou": miou,
            "mask_dice": mdice,
            "gt_area_px": gsum,
            "pred_area_px": psum,
            "inter_area_px": inter,
            "mean_match_center_error_px": float(np.mean([m["center_distance"] for m in matches])) if matches else None,
            "mean_match_circle_iou": float(np.mean([m["circle_iou"] for m in matches])) if matches else None,
            "matched_solder_for_rate": len(solder_matches),
            "void_rate_mae": float(np.mean(image_rate_errors)) if image_rate_errors else None,
            "ng_accuracy": safe_div(image_ng_correct, len(solder_matches)) if solder_matches else None,
            "infer_eval_seconds": infer_s,
        }

        totals["gt"] += len(gt_eval)
        totals["pred"] += len(pred_holes)
        totals["tp"] += tp_count
        totals["fp"] += fp_count
        totals["fn"] += fn_count

        # 保存可视化对比标注图 (1:1 单张原图尺寸，纯圆圈无内部十字)
        vis = draw_eval(
            img, gt_eval, pred_holes, matches, unmatched_gt, unmatched_pred, metrics,
            gt_solders=gt_solders,
            pred_solders=pred_solders,
            unmatched_gt_solders=unmatched_gt_solders,
            unmatched_pred_solders=unmatched_pred_solders,
            line_thickness=args.line_thickness
        )
        vis_path = output / "visualizations" / f"{ip.stem}_void_eval.jpg"
        imwrite_unicode(vis_path, vis)

        # 保存单图详细 JSON
        detail = {
            "metrics": metrics,
            "gt_solders": [asdict(x) for x in gt_solders],
            "gt_holes_evaluated": [asdict(x) for x in gt_eval],
            "pred_solders": [asdict(x) for x in pred_solders],
            "pred_holes": [asdict(x) for x in pred_holes],
            "matches": matches,
            "unmatched_gt_indices": unmatched_gt,
            "unmatched_pred_indices": unmatched_pred,
            "unmatched_gt_solders": unmatched_gt_solders,
            "unmatched_pred_solders": unmatched_pred_solders,
        }
        with (output / "per_image" / f"{ip.stem}.json").open("w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)

        rows.append(metrics)
        print(
            f"[{idx:2d}/{len(pairs):2d}] {ip.name:<8} | "
            f"GT-hole={len(gt_eval):3d} Pred-hole={len(pred_holes):3d} | "
            f"TP={tp_count:3d} FP={fp_count:3d} FN={fn_count:3d} | "
            f"面积Recall={recall_area*100:5.1f}% (数量={recall_count*100:5.1f}%) "
            f"面积P={precision_area*100:5.1f}% (数量={precision_count*100:5.1f}%) | "
            f"耗时={infer_s:.2f}s"
        )

    micro_p_count = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    micro_r_count = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    micro_f1_count = f1_score(micro_p_count, micro_r_count)
    micro_acc_count = safe_div(totals["tp"], totals["tp"] + totals["fp"] + totals["fn"])

    global_mask_iou = safe_div(mask_inters, mask_unions)
    global_mask_dice = safe_div(2 * mask_inters, mask_gt_sum + mask_pred_sum)
    global_area_recall = safe_div(mask_inters, mask_gt_sum)
    global_area_precision = safe_div(mask_inters, mask_pred_sum)
    global_area_f1 = global_mask_dice
    global_area_accuracy = global_mask_iou
    global_circle_area_recall = safe_div(total_matched_gt_circle_area, total_gt_circle_area)

    macro = lambda key: float(np.mean([float(r[key]) for r in rows])) if rows else 0.0

    # 构造全数据集合计行 (特别包含总体面积召回率与汇总指标)
    summary_row = {
        "image": "全数据集合计",
        "json": f"共 {len(rows)} 张图",
        "width": "-",
        "height": "-",
        "size_note": f"宏平均AreaRecall:{macro('recall_area')*100:.2f}%",
        "gt_solder_count": sum(int(r.get("gt_solder_count", 0)) for r in rows),
        "gt_count": totals["gt"],
        "pred_solder_count": sum(int(r.get("pred_solder_count", 0)) for r in rows),
        "pred_count": totals["pred"],
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": global_area_precision,
        "recall": global_area_recall,  # 🌟 全局总体面积召回率
        "f1": global_area_f1,
        "accuracy": global_area_accuracy,
        "recall_area": global_area_recall,
        "recall_circle_area": global_circle_area_recall,
        "recall_count": micro_r_count,
        "precision_area": global_area_precision,
        "precision_count": micro_p_count,
        "f1_area": global_area_f1,
        "f1_count": micro_f1_count,
        "accuracy_area": global_area_accuracy,
        "accuracy_count": micro_acc_count,
        "mask_iou": global_mask_iou,
        "mask_dice": global_mask_dice,
        "gt_area_px": mask_gt_sum,
        "pred_area_px": mask_pred_sum,
        "inter_area_px": mask_inters,
        "mean_match_center_error_px": float(np.mean(matched_center_errors)) if matched_center_errors else "",
        "mean_match_circle_iou": float(np.mean(matched_circle_ious)) if matched_circle_ious else "",
        "matched_solder_for_rate": total_matched_solders_for_rate,
        "void_rate_mae": float(np.mean(rate_abs_errors)) if rate_abs_errors else "",
        "ng_accuracy": safe_div(ng_correct, ng_total) if ng_total else "",
        "infer_eval_seconds": sum(float(r.get("infer_eval_seconds", 0)) for r in rows),
    }

    # 汇总输出 metrics.csv (包含各图明细与最后一行全数据集合计)
    fields = [
        "image", "json", "width", "height", "size_note",
        "gt_solder_count", "gt_count", "pred_solder_count", "pred_count",
        "tp", "fp", "fn",
        "recall", "precision", "f1", "accuracy",
        "recall_area", "recall_circle_area", "recall_count",
        "precision_area", "precision_count", "f1_area", "f1_count",
        "accuracy_area", "accuracy_count",
        "mask_iou", "mask_dice", "gt_area_px", "pred_area_px", "inter_area_px",
        "mean_match_center_error_px", "mean_match_circle_iou",
        "matched_solder_for_rate", "void_rate_mae", "ng_accuracy", "infer_eval_seconds",
    ]
    write_csv(output / "metrics.csv", rows + [summary_row], fields)

    summary = {
        "images": len(rows),
        "gt": totals["gt"],
        "pred": totals["pred"],
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "micro_precision_area": global_area_precision,
        "micro_recall_area": global_area_recall,
        "micro_f1_area": global_area_f1,
        "micro_accuracy_area": global_area_accuracy,
        "micro_recall_circle_area": global_circle_area_recall,
        "micro_precision_count": micro_p_count,
        "micro_recall_count": micro_r_count,
        "micro_f1_count": micro_f1_count,
        "micro_accuracy_count": micro_acc_count,
        "macro_precision_area": macro("precision_area"),
        "macro_recall_area": macro("recall_area"),
        "macro_f1_area": macro("f1_area"),
        "macro_recall_circle_area": macro("recall_circle_area"),
        "macro_recall_count": macro("recall_count"),
        "macro_precision_count": macro("precision_count"),
        "macro_f1_count": macro("f1_count"),
        "global_mask_iou": global_mask_iou,
        "global_mask_dice": global_mask_dice,
        "gt_area_px": mask_gt_sum,
        "pred_area_px": mask_pred_sum,
        "inter_area_px": mask_inters,
        "mean_matched_center_error_px": float(np.mean(matched_center_errors)) if matched_center_errors else None,
        "mean_matched_circle_iou": float(np.mean(matched_circle_ious)) if matched_circle_ious else None,
        "matched_solders_for_void_rate": total_matched_solders_for_rate,
        "void_rate_mae": float(np.mean(rate_abs_errors)) if rate_abs_errors else None,
        "ng_accuracy": safe_div(ng_correct, ng_total) if ng_total else None,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    try:
        with (output / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except PermissionError:
        alt_sum = output / f"summary_{int(time.time())}.json"
        print(f"\n[提示] summary.json 被占用锁定，已自动另存为: {alt_sum.name}")
        with alt_sum.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("                评估结果汇总 (Summary)")
    print("=" * 60)
    print(f"评估图像总数         : {summary['images']} 张")
    print(f"GT 总孔洞 / 预测孔洞 : {totals['gt']} / {totals['pred']}")
    print(f"TP / FP / FN (数量)  : {totals['tp']} / {totals['fp']} / {totals['fn']}")
    print(f"🌟 总体面积 Recall   : {global_area_recall * 100:.2f}%  (宏平均: {summary['macro_recall_area'] * 100:.2f}%)")
    print(f"🌟 目标圆加权 Recall : {global_circle_area_recall * 100:.2f}%  (检出真值圆面积占比)")
    print(f"🌟 总体数量 Recall   : {micro_r_count * 100:.2f}%  (宏平均: {summary['macro_recall_count'] * 100:.2f}%)")
    print(f"🌟 总体面积 Precision: {global_area_precision * 100:.2f}%  (宏平均: {summary['macro_precision_area'] * 100:.2f}%)")
    print(f"🌟 总体面积 F1 (Dice): {global_area_f1 * 100:.2f}%  (宏平均: {summary['macro_f1_area'] * 100:.2f}%)")
    print(f"全局 Mask IoU        : {global_mask_iou * 100:.2f}%")
    if summary["void_rate_mae"] is not None:
        print(f"孔洞率绝对误差 (MAE) : {summary['void_rate_mae'] * 100:.2f}%")
    if summary["ng_accuracy"] is not None:
        print(f"NG 判定准确率        : {summary['ng_accuracy'] * 100:.2f}%")
    print("-" * 60)
    print(f"报表 CSV             : {output / 'metrics.csv'}")
    print(f"结果 JSON            : {output / 'summary.json'}")
    print(f"可视化图像目录       : {output / 'visualizations'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
