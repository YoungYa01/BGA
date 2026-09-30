# -*- coding: utf-8 -*-
"""
🌟 BGA 焊点间桥连缺陷高精度检测模块 (BGA Bridge Detection Module)
------------------------------------------------------------------
设计定位:
1. 独立纯算法模块，与已有焊点定位/气泡检测解耦；
2. 严格紧密对齐工业金标准标注规则：
   - 长方形 ROI 检测区: 长度为两圆心距离，宽度紧密贴合焊球直径 (23~26px)；
   - 实心小锡球测谎: 尺寸小于主焊球，具有实心金属厚度膨胀与暗色灰度特征；
   - 杜绝误检: 排除过孔 (Via)、长方形贴片阻容器件 (SMD)、PCB 细走线与侧面铜箔；
   - 双端边缘连接判定: 小锡球边缘必须同时连通两焊球边缘 (只连单侧坚决不报)。
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

# 保证在 Windows 控制台下正常输出 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ==============================================================
# ================= 🌟 [数据模型定义] 对齐接口规范 =================
# ==============================================================
class BridgeDefect(TypedDict):
    roi_box: Tuple[int, int, int, int]    # 桥连长方形检测框 (x, y, w, h)
    small_ball: Tuple[int, int, int]      # 桥连小锡球 (center_x, center_y, radius)
    ball_A: Dict[str, Any]                # 相邻焊球 A 信息
    ball_B: Dict[str, Any]                # 相邻焊球 B 信息
    orientation: str                      # 走向: "H" (水平) 或 "V" (垂直)
    confidence: float                     # 缺陷置信度: 0.0 ~ 1.0


COLOR_BRIDGE_BOX = (0, 0, 255)    # 桥连长方形检测框颜色 (红色)
COLOR_BRIDGE_BALL = (0, 255, 255) # 桥连小锡球标记圆颜色 (黄色)


def _normalize_solder_points(solder_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """统一焊球字典字段名称 (兼容 detector.py、bga_void_seg.py 与标准坐标 json)"""
    normalized = []
    for idx, p in enumerate(solder_points):
        cx = float(p.get("CenterX", p.get("center_x", p.get("x", 0.0))))
        cy = float(p.get("CenterY", p.get("center_y", p.get("y", 0.0))))

        draw_r = float(p.get("draw_radius", p.get("DrawRadius", 0.0)) or 0.0)
        raw_r = float(p.get("radius", p.get("Radius", 0.0)) or 0.0)

        if 5.0 <= draw_r <= 25.0:
            r = draw_r
        elif raw_r > 0.0:
            r = raw_r
        else:
            w = float(p.get("w", p.get("Width", 0.0)) or 0.0)
            h = float(p.get("h", p.get("Height", 0.0)) or 0.0)
            r = (w + h) / 4.0 if (w > 0 and h > 0) else 0.0

        if r >= 3.5:
            normalized.append({
                "x": cx,
                "y": cy,
                "r": r,
                "orig_index": p.get("orig_index", idx)
            })
    return normalized


def detect_bga_bridges(
    image: np.ndarray,
    solder_points: List[Dict[str, Any]],
    result_img: Optional[np.ndarray] = None,
    draw_on_result: bool = True,
) -> List[BridgeDefect]:
    """
    🌟 BGA 焊点间桥连缺陷高精度检测纯函数
    
    参数:
        image: 输入原始图像 (BGR 或灰度图)
        solder_points: 基准焊球列表 (由 YOLO/标注提供，包含圆心坐标与半径)
        result_img: (可选) 可视化图层，若提供且 draw_on_result=True 则在其上绘制红框与黄圆
        draw_on_result: 是否在 result_img 上绘制桥连标记
    
    返回:
        检出的 BridgeDefect 缺陷列表
    """
    if image is None or len(solder_points) < 2:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    h_img, w_img = gray.shape[:2]

    norm_points = _normalize_solder_points(solder_points)
    if len(norm_points) < 2:
        return []

    # =========================================================================
    # 🌟 阶段 1：前置清洗——区分真实实心金属焊球与空心高亮过孔 (用户需求 4)
    # =========================================================================
    balls = [] # 真实深色金属实心焊球
    vias = []  # 板子上的空心高亮过孔 (中心透亮发白)

    for p in norm_points:
        cx, cy, ir = int(round(p["x"])), int(round(p["y"])), int(round(p["r"]))

        # 测定焊球/过孔核心灰度: 过孔穿透孔中心在 X 射线图像中呈极亮高光
        mask_core = np.zeros_like(gray)
        cv2.circle(mask_core, (cx, cy), max(2, int(ir * 0.35)), 255, -1)
        core_mean_val = cv2.mean(gray, mask=mask_core)[0]
        core_max_val = np.max(gray[mask_core > 0]) if np.any(mask_core > 0) else 0

        # 需求 4: 过孔拦截 (中心极亮 > 165 或峰值 > 220 即为穿透圆孔，绝不作为主焊球)
        if core_mean_val > 165 or core_max_val > 220:
            vias.append({"x": p["x"], "y": p["y"], "r": p["r"]})
            continue

        # 真实焊球实体高精度轮廓提取 (半径严格受控，确切包裹金属实体，绝不虚高吞噬外部桥连)
        pad = int(ir * 1.2)
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(w_img, cx + pad), min(h_img, cy + pad)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
        _, bin_s = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bin_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        real_cx, real_cy, real_r = p["x"], p["y"], float(ir)
        if cnts:
            roi_mid = (roi.shape[1] / 2.0, roi.shape[0] / 2.0)
            c_best = None
            min_d = float("inf")
            for c in cnts:
                if cv2.contourArea(c) < 15:
                    continue
                (tcx, tcy), tr = cv2.minEnclosingCircle(c)
                d_c = math.hypot(tcx - roi_mid[0], tcy - roi_mid[1])
                if d_c < min_d:
                    min_d = d_c
                    c_best = (tcx, tcy, tr)
            if c_best and min_d < ir * 0.5:
                tcx, tcy, tr = c_best
                fit_r = min(tr * 0.98, float(ir))
                if fit_r >= 4:
                    real_cx = tcx + x1
                    real_cy = tcy + y1
                    real_r = fit_r

        balls.append({
            "x": real_cx,
            "y": real_cy,
            "r": real_r,
            "core": core_mean_val,
            "orig_index": p["orig_index"],
        })

    if len(balls) < 2:
        return []

    # 全局二值图与连通域分析 (用于超大贴片器件/铜箔长方形干扰拦截)
    blur_full = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_full = cv2.threshold(blur_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels_full, labels_full, stats_full, _ = cv2.connectedComponentsWithStats(bin_full)

    ball_radii = [b["r"] for b in balls]
    typical_r = float(np.median(ball_radii)) if ball_radii else 12.0
    typical_ball_area = math.pi * (typical_r**2)

    # =========================================================================
    # 🌟 阶段 2：自适应间距 (Pitch) 估算与相邻焊球对配对
    # =========================================================================
    min_dists = []
    for i in range(len(balls)):
        p1 = (balls[i]["x"], balls[i]["y"])
        md = min(math.hypot(p1[0] - balls[j]["x"], p1[1] - balls[j]["y"]) for j in range(len(balls)) if i != j)
        min_dists.append(md)
    min_dists.sort()
    pitch = min_dists[len(min_dists) // 2] if min_dists else 30.0

    pairs_to_check = []
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            b1, b2 = balls[i], balls[j]
            dx = abs(b1["x"] - b2["x"])
            dy = abs(b1["y"] - b2["y"])
            d = math.hypot(dx, dy)

            # 只检测空间距离在 0.65 ~ 1.45 倍标准间距的相邻球
            if not (0.65 * pitch <= d <= 1.45 * pitch):
                continue

            is_horizontal = (dx > dy * 1.3) and (dy < 0.40 * pitch)
            is_vertical = (dy > dx * 1.3) and (dx < 0.40 * pitch)

            if is_horizontal or is_vertical:
                pairs_to_check.append((b1, b2, "H" if is_horizontal else "V", d))

    # =========================================================================
    # 🌟 阶段 3：长方形 ROI 内实心小锡球双端边缘连接检测 (核心质检流水线)
    # =========================================================================
    detected_defects: List[BridgeDefect] = []

    for ball_A, ball_B, orient, dist_ab in pairs_to_check:
        # 🚀 规则 1：规避背景超大长方形干扰项（贴片电容、电阻、芯片、大铜箔）(需求 2)
        ix1, iy1 = int(round(ball_A["x"])), int(round(ball_A["y"]))
        ix2, iy2 = int(round(ball_B["x"])), int(round(ball_B["y"]))
        lbl_full1 = labels_full[min(max(0, iy1), h_img - 1), min(max(0, ix1), w_img - 1)]
        lbl_full2 = labels_full[min(max(0, iy2), h_img - 1), min(max(0, ix2), w_img - 1)]

        if lbl_full1 > 0 and lbl_full1 == lbl_full2:
            comp_area = stats_full[lbl_full1, cv2.CC_STAT_AREA]
            comp_w = stats_full[lbl_full1, cv2.CC_STAT_WIDTH]
            comp_h = stats_full[lbl_full1, cv2.CC_STAT_HEIGHT]
            # 过滤超大贴片器件或大面积覆铜
            if comp_area > typical_ball_area * 5.0:
                continue
            if comp_w > pitch * 3.5 or comp_h > pitch * 3.5:
                continue
            # 过滤标准长方形贴片器件 (矩形饱满度极高)
            if float(comp_area) / max(1, comp_w * comp_h) > 0.85:
                continue

        max_r = max(ball_A["r"], ball_B["r"])
        mid_x = (ball_A["x"] + ball_B["x"]) / 2.0
        mid_y = (ball_A["y"] + ball_B["y"]) / 2.0

        # 🚀 规则 2：需求 5——长方形几何规范 (紧密对齐金标准标注尺寸)
        # 依据点阵特征间距 (Pitch) 自适应判定大焊球阵列 (pitch > 36px) 与小焊球阵列 (pitch <= 36px)
        # 大焊球 ROI 宽 26px，小焊球 ROI 宽 23px，在两侧留出充分安全距离，坚决杜绝侵入过孔！
        is_large_pitch = (pitch > 36.0)
        box_width = 26.0 if is_large_pitch else 23.0
        half_w = box_width / 2.0

        if orient == "H":
            x1_box = int(round(min(ball_A["x"], ball_B["x"])))
            x2_box = int(round(max(ball_A["x"], ball_B["x"])))
            y1_box = int(round(mid_y - half_w))
            y2_box = int(round(mid_y + half_w))
        else:
            x1_box = int(round(mid_x - half_w))
            x2_box = int(round(mid_x + half_w))
            y1_box = int(round(min(ball_A["y"], ball_B["y"])))
            y2_box = int(round(max(ball_A["y"], ball_B["y"])))

        x1_box, y1_box = max(0, x1_box), max(0, y1_box)
        x2_box, y2_box = min(w_img, x2_box), min(h_img, y2_box)
        if x2_box - x1_box < 6 or y2_box - y1_box < 6:
            continue

        roi_gray = gray[y1_box:y2_box, x1_box:x2_box]
        h_roi, w_roi = roi_gray.shape[:2]

        loc_ax, loc_ay = int(round(ball_A["x"] - x1_box)), int(round(ball_A["y"] - y1_box))
        loc_bx, loc_by = int(round(ball_B["x"] - x1_box)), int(round(ball_B["y"] - y1_box))

        # 局部大津法二值化提取深色金属实体
        blur_roi = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        _, bin_metal = cv2.threshold(blur_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 🚀 规则 3：连通性预检——两端焊球必须存在物理连通通路
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bin_metal, connectivity=8)
        lbl_A = labels[min(max(0, loc_ay), h_roi - 1), min(max(0, loc_ax), w_roi - 1)]
        lbl_B = labels[min(max(0, loc_by), h_roi - 1), min(max(0, loc_bx), w_roi - 1)]
        if lbl_A == 0 or lbl_B == 0 or lbl_A != lbl_B:
            continue

        # 两端焊球边缘环带 (用于双端连接判定，严格贴合球外缘)
        ring_A = np.zeros_like(bin_metal)
        cv2.circle(ring_A, (loc_ax, loc_ay), max(2, int(round(ball_A["r"] * 1.05))), 255, 2)
        ring_B = np.zeros_like(bin_metal)
        cv2.circle(ring_B, (loc_bx, loc_by), max(2, int(round(ball_B["r"] * 1.05))), 255, 2)

        # 精确擦除两端主焊球 (1.02 * r)，保留缝隙中间的小锡球主体与边缘连接
        bin_gap = bin_metal.copy()
        cv2.circle(bin_gap, (loc_ax, loc_ay), int(round(ball_A["r"] * 1.02)), 0, -1)
        cv2.circle(bin_gap, (loc_bx, loc_by), int(round(ball_B["r"] * 1.02)), 0, -1)
        bin_gap = cv2.morphologyEx(bin_gap, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        contours, _ = cv2.findContours(bin_gap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        ref_ball_area = math.pi * (((ball_A["r"] + ball_B["r"]) / 2.0)**2)
        is_bridged = False
        bridge_center = None
        bridge_radius = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 过滤 A：面积约束 (排除微小噪点与超大贴片器件)
            if area < 18 or area > ref_ball_area * 0.95:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # 🚀 过滤 B：规避长方形贴片元件与走线干扰项 (需求 2、3)
            # B.1 偏心度检测: 桥连小球居于两球正中主轴缝隙，排除偏心过大的侧面走线
            (cx_c, cy_c), cr = cv2.minEnclosingCircle(cnt)
            c_dist_from_axis = abs(cx_c - w_roi / 2.0) if orient == "V" else abs(cy_c - h_roi / 2.0)
            axis_tol = 0.38 if is_large_pitch else 0.22
            if c_dist_from_axis > box_width * axis_tol:
                continue

            # B.2 长宽比约束: 小锡球是圆润凸形，排除扁平切线与极长平行走线
            aspect_ratio = bw / float(bh) if bh > 0 else 0
            if aspect_ratio < 0.35 or aspect_ratio > 2.8:
                continue

            # B.3 旋转矩形度与直线度过滤: 拦截标准长方形贴片电阻/电容
            rot_rect = cv2.minAreaRect(cnt)
            rw, rh = rot_rect[1]
            if rw * rh > 0:
                rect_ratio = area / (rw * rh)
                rot_aspect = max(rw, rh) / max(1.0, min(rw, rh))
                if rect_ratio > 0.86 and rot_aspect > 1.6:
                    continue
                if rot_aspect > 2.4:
                    continue

            # 🚀 过滤 C：实心圆球厚度与几何形态约束
            cnt_mask = np.zeros_like(bin_gap)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            dist_map = cv2.distanceTransform(cnt_mask, cv2.DIST_L2, 3)
            dist_peak = float(dist_map.max())
            # 实心小圆球必须有物理厚度膨胀 (内切圆半径峰值 >= 2.5 像素，彻底杜绝细走线)
            if dist_peak < 2.5:
                continue

            # 需求 3：桥连小锡球尺寸必须小于主焊点
            if cr > min(ball_A["r"], ball_B["r"]) * 0.98:
                continue

            # 圆度测谎 (大焊球大间距放宽至 0.36，小焊球紧凑排布要求 0.58)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            min_circ = 0.36 if is_large_pitch else 0.58
            if circularity < min_circ:
                continue

            # 实体凸包饱满度测谎 (排除过度凹陷的空壳或噪点)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            if solidity < 0.65:
                continue

            # 🚀 过滤 D：需求 4——规避板子上的过孔 (过孔中心高光透亮，金属小球深暗)
            core_mask = np.zeros_like(roi_gray)
            cv2.circle(core_mask, (int(round(cx_c)), int(round(cy_c))), max(2, int(round(cr * 0.35))), 255, -1)
            core_gray = cv2.mean(roi_gray, mask=core_mask)[0]
            if core_gray > 130:
                continue

            # 过孔空间距离测谎: 小球中心不得与任何已知过孔中心空间重合
            global_cx = cx_c + x1_box
            global_cy = cy_c + y1_box
            if any(math.hypot(global_cx - v["x"], global_cy - v["y"]) < 8.0 for v in vias):
                continue

            # 🚀 过滤 E：需求 5——双端边缘连接判定 (最核心要求: 只连单侧不算，必须两端同时连通)
            touch_k = max(3, int(max_r * 0.25))
            if touch_k % 2 == 0:
                touch_k += 1
            touched = cv2.dilate(cnt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (touch_k, touch_k)))

            touch_A = np.any(cv2.bitwise_and(touched, ring_A))
            touch_B = np.any(cv2.bitwise_and(touched, ring_B))

            # 必须同时连接两焊点的边缘！
            if touch_A and touch_B:
                is_bridged = True
                bridge_center = (int(round(global_cx)), int(round(global_cy)))
                bridge_radius = max(3, int(round(cr)))
                break

        if is_bridged and bridge_center is not None:
            rw_box = x2_box - x1_box
            rh_box = y2_box - y1_box
            defect: BridgeDefect = {
                "roi_box": (x1_box, y1_box, rw_box, rh_box),
                "small_ball": (bridge_center[0], bridge_center[1], bridge_radius),
                "ball_A": {"center": (ball_A["x"], ball_A["y"]), "radius": ball_A["r"], "index": ball_A.get("orig_index", -1)},
                "ball_B": {"center": (ball_B["x"], ball_B["y"]), "radius": ball_B["r"], "index": ball_B.get("orig_index", -1)},
                "orientation": orient,
                "confidence": 0.95,
            }
            detected_defects.append(defect)

            # 可选绘制
            if draw_on_result and result_img is not None:
                cv2.rectangle(result_img, (x1_box, y1_box), (x2_box, y2_box), COLOR_BRIDGE_BOX, 1)
                cv2.circle(result_img, bridge_center, bridge_radius, COLOR_BRIDGE_BALL, 1)

    return detected_defects
