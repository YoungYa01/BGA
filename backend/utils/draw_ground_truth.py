#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGA 原始人工标注可视化绘制工具 (Ground Truth Visualizer)

功能：
1. 自动读取 data/new-img (原始图像) 和 data/new-coord (LabelMe 标注 JSON)；
2. 解析人工标注的焊球 (solder ball) 与气泡 (hole) 坐标及半径；
3. 支持坐标自适应尺寸缩放与去重；
4. 采用精细抗锯齿细线将人工真实标注 1:1 绘制在原图上；
5. 在左上角添加紧凑信息栏与图例 (Legend)；
6. 自动保存标注图至 output/void_eval_results/origin 目录。
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

HERE = Path(__file__).resolve().parent

# 支持的常见图像扩展名
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# 颜色配置 (BGR 格式)
COLOR_SOLDER_GT = (220, 220, 70)   # 人工标注焊球圈：淡青色
COLOR_HOLE_GT = (0, 165, 255)      # 人工标注气泡圈：明亮橙色 (与评估系统的 FN 橙色保持一致)
COLOR_TEXT = (245, 245, 245)       # 信息文字：白灰
COLOR_BG_BOX = (20, 22, 26)        # 顶部信息栏背景：深灰黑色
COLOR_BG_BORDER = (60, 65, 70)     # 信息栏边框：浅灰


def imread_unicode(path: Path) -> np.ndarray:
    """读取支持中文路径的图像"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """保存支持中文路径的图像"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    buf.tofile(str(path))


def parse_labelme_circles(data: Dict[str, Any], label: str) -> List[Tuple[float, float, float]]:
    """解析 LabelMe JSON 中的圆标注: 返回 [(cx, cy, r), ...]"""
    circles = []
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
        circles.append((x0, y0, r))
    return circles


def deduplicate_circles(circles: Sequence[Tuple[float, float, float]], dist_thresh: float = 3.0) -> List[Tuple[float, float, float]]:
    """消除标注数据中偶发的重复标注圈 (圆心距离小于 dist_thresh 视为重复)"""
    clean: List[Tuple[float, float, float]] = []
    for c in circles:
        if not any(math.hypot(c[0] - u[0], c[1] - u[1]) < dist_thresh for u in clean):
            clean.append(c)
    return clean


def scale_circles(circles: Sequence[Tuple[float, float, float]], sx: float, sy: float) -> List[Tuple[float, float, float]]:
    """按比例缩放圆坐标"""
    sr = math.sqrt(max(1e-12, sx * sy))
    return [(c[0] * sx, c[1] * sy, c[2] * sr) for c in circles]


def draw_ground_truth_image(
    image: np.ndarray,
    image_name: str,
    solders: Sequence[Tuple[float, float, float]],
    holes: Sequence[Tuple[float, float, float]],
    line_thickness: int = 1,
    draw_center_dot: bool = False
) -> np.ndarray:
    """在图像上绘制人工标注的焊球和气泡，并添加图例信息"""
    h, w = image.shape[:2]
    vis = image.copy()

    # 1. 绘制人工标注的焊球外接圆 (淡青色)
    for x, y, r in solders:
        cx, cy, cr = int(round(x)), int(round(y)), max(1, int(round(r)))
        cv2.circle(vis, (cx, cy), cr, COLOR_SOLDER_GT, line_thickness, cv2.LINE_AA)
        if draw_center_dot:
            cv2.circle(vis, (cx, cy), 1, COLOR_SOLDER_GT, -1, cv2.LINE_AA)

    # 2. 绘制人工标注的气泡圆 (明亮橙色)
    for x, y, r in holes:
        cx, cy, cr = int(round(x)), int(round(y)), max(1, int(round(r)))
        cv2.circle(vis, (cx, cy), cr, COLOR_HOLE_GT, line_thickness, cv2.LINE_AA)
        if draw_center_dot:
            cv2.circle(vis, (cx, cy), 1, COLOR_HOLE_GT, -1, cv2.LINE_AA)

    # 3. 顶部紧凑信息栏与图例
    info_text = f"[{image_name}]  GT Solder Balls: {len(solders)} | GT Holes (Voids): {len(holes)} | Size: {w}x{h}"
    box_w = min(w - 12, 700)
    box_h = 44
    cv2.rectangle(vis, (6, 6), (6 + box_w, 6 + box_h), COLOR_BG_BOX, -1)
    cv2.rectangle(vis, (6, 6), (6 + box_w, 6 + box_h), COLOR_BG_BORDER, 1)

    # 第一行: 统计文本
    cv2.putText(vis, info_text, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_TEXT, 1, cv2.LINE_AA)

    # 第二行: 图例 (Legend)
    lg_y = 42
    # 焊球图例
    cv2.circle(vis, (18, lg_y - 4), 5, COLOR_SOLDER_GT, line_thickness + 1, cv2.LINE_AA)
    cv2.putText(vis, "GT Solder Ball (Annotated)", (30, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_SOLDER_GT, 1, cv2.LINE_AA)
    # 气泡图例
    cv2.circle(vis, (250, lg_y - 4), 5, COLOR_HOLE_GT, line_thickness + 1, cv2.LINE_AA)
    cv2.putText(vis, "GT Void / Hole (Annotated)", (262, lg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_HOLE_GT, 1, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="将人工标注的 BGA 焊球与气泡绘制在原图中")
    parser.add_argument("--img-dir", default=str(HERE / "data" / "new-img"), help="图片目录路径 (默认: data/new-img)")
    parser.add_argument("--coord-dir", default=str(HERE / "data" / "new-coord"), help="标注 JSON 目录路径 (默认: data/new-coord)")
    parser.add_argument("--output", default=str(HERE / "output" / "void_eval_results" / "ground-truth"), help="输出目录 (默认: output/void_eval_results/ground-truth)")
    parser.add_argument("--solder-label", default="solder ball", help="焊球标签名称 (默认: solder ball)")
    parser.add_argument("--hole-label", default="hole", help="孔洞标签名称 (默认: hole)")
    parser.add_argument("--line-thickness", type=int, default=1, help="圆圈线条粗细 (默认: 1)")
    parser.add_argument("--center-dot", action="store_true", help="是否在圆心绘制微小中心点")
    args = parser.parse_args()

    img_dir = Path(args.img_dir).resolve()
    coord_dir = Path(args.coord_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"❌ 图像目录不存在: {img_dir}")
        return 1
    if not coord_dir.exists():
        print(f"❌ 标注目录不存在: {coord_dir}")
        return 1

    # 扫描图像与 JSON 配对
    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    pairs = []
    for ip in img_files:
        jp = coord_dir / f"{ip.stem}.json"
        if jp.exists():
            pairs.append((ip, jp))

    if not pairs:
        print("⚠️ 未找到任何成对的图像与 JSON 文件！")
        return 1

    print("=" * 65)
    print("        BGA 人工标注真实图绘制工具 (Ground Truth Visualizer)")
    print("=" * 65)
    print(f"图像目录   : {img_dir}")
    print(f"标注目录   : {coord_dir}")
    print(f"输出目录   : {output_dir}")
    print(f"有效标注对 : {len(pairs)} 组")
    print("-" * 65)

    total_solders = 0
    total_holes = 0

    for idx, (ip, jp) in enumerate(pairs, 1):
        img = imread_unicode(ip)
        if img is None:
            print(f"[{idx:2d}/{len(pairs):2d}] ❌ 读取图片失败: {ip.name}")
            continue

        h, w = img.shape[:2]

        with jp.open("r", encoding="utf-8-sig") as f:
            jd = json.load(f)

        solders = deduplicate_circles(parse_labelme_circles(jd, args.solder_label))
        holes = deduplicate_circles(parse_labelme_circles(jd, args.hole_label))

        # 检查是否需要按原图尺寸比例缩放标注
        jw = int(jd.get("imageWidth") or 0)
        jh = int(jd.get("imageHeight") or 0)
        if jw > 0 and jh > 0 and (jw != w or jh != h):
            sx, sy = w / float(jw), h / float(jh)
            solders = scale_circles(solders, sx, sy)
            holes = scale_circles(holes, sx, sy)

        # 绘制标注图像
        vis = draw_ground_truth_image(
            img, ip.name, solders, holes,
            line_thickness=args.line_thickness,
            draw_center_dot=args.center_dot
        )

        out_path = output_dir / f"{ip.stem}_origin.jpg"
        imwrite_unicode(out_path, vis)

        total_solders += len(solders)
        total_holes += len(holes)

        print(f"[{idx:2d}/{len(pairs):2d}] 已保存: {out_path.name:<18} | 焊球: {len(solders):3d} | 气泡: {len(holes):3d} | 尺寸: {w}x{h}")

    print("-" * 65)
    print(f"🎉 全部完成！共处理 {len(pairs)} 张图像，合计焊球: {total_solders} 个，气泡: {total_holes} 个。")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
