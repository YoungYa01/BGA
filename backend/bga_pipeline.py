# -*- coding: utf-8 -*-
"""
🌟 BGA 一体化工业质检流水线系统 (BGA Inspection Pipeline)
-------------------------------------------------------------
支持三种清晰独立的质检维度 (模式)：
1. 气泡质检 (Void Only)         : inspect_bga_void()
2. 桥连质检 (Bridge Only)       : inspect_bga_bridge()
3. 综合质检 (Comprehensive / All): inspect_bga_comprehensive() [或 inspect_bga()]
"""
from __future__ import annotations

import os
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

# 保证在 Windows 控制台下正常输出 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 引入已有检测器与气泡分割模块
from detector import BGADetector, result_to_objects
from utils.grid_node_refiner import (
    GridNodeRefineConfig,
    apply_grid_node_refine_to_result_item,
)
from bga_void_seg import (
    PredictionResult,
    get_detector_instance,
    predict_and_generate_mask,
    COLOR_SOLDER,
    COLOR_VOID_NG,
    COLOR_VOID_PASS,
    COLOR_TEXT_NG,
    COLOR_TEXT_PASS,
    GRID_REFINE_CFG,
    SOLDER_MIN_RADIUS,
)
from bga_bridge_detect import (
    BridgeDefect,
    detect_bga_bridges,
    COLOR_BRIDGE_BOX,
    COLOR_BRIDGE_BALL,
)


# ==============================================================
# ================= 🌟 [质检模式枚举与数据模型] ==================
# ==============================================================
class InspectionMode(str, Enum):
    VOID = "void"                   # 仅气泡检测
    BRIDGE = "bridge"               # 仅桥连检测
    COMPREHENSIVE = "comprehensive" # 全项综合检测 (气泡 + 桥连)


class BoardInspectionResult(TypedDict):
    mode: str                             # 质检模式: "void" | "bridge" | "comprehensive"
    board_status: str                     # 整板判定: "PASS" 或 "NG"
    ng_reasons: List[str]                 # 不合格原因列表
    summary: Dict[str, Any]               # 汇总统计数据
    void_details: List[PredictionResult]  # 各焊球气泡详细信息 (仅气泡/综合模式提供)
    bridge_details: List[BridgeDefect]    # 桥连缺陷详细信息 (仅桥连/综合模式提供)
    visual_output_path: Optional[str]     # 最终合成质检标注图路径


# ==============================================================
# ================= 🌟 [核心质检功能 1: 仅气泡质检] =============
# ==============================================================
def inspect_bga_void(
    input_image_path: str,
    weights_path: str,
    conf_threshold: float = 0.50,
    ng_void_threshold: float = 0.25,     # 气泡超标门槛 (默认 25%)
    device: str = "0",
    save_debug_image: bool = True,
    debug_output_dir: Optional[str] = None,
) -> BoardInspectionResult:
    """
    🔍 功能一：BGA 焊球【仅气泡/空洞率】质检 (Void Only Inspection)
    - 专注定位焊球并精确计算单球内部空洞率 (Void Rate)；
    - 不耗费算力进行相邻焊球桥连巡检；
    - 整板判定标准：无任何焊球气泡率超过 ng_void_threshold 则 PASS，否则 NG。
    """
    t_start = time.time()
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"未找到输入图片: {input_image_path}")

    img = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片数据: {input_image_path}")
    vis_img = img.copy() if save_debug_image else None

    # 调用底层气泡分割 (关闭内部写盘，由流水线统一渲染)
    void_results: List[PredictionResult] = predict_and_generate_mask(
        model=weights_path,
        input_image_path=input_image_path,
        conf_threshold=conf_threshold,
        ng_threshold=ng_void_threshold * 100.0,
        device=device,
        save_debug_image=False,
    )

    ng_reasons: List[str] = []
    void_ng_count = 0
    max_void_rate = 0.0

    for idx, item in enumerate(void_results):
        vr = item["void_rate"]
        if vr > max_void_rate:
            max_void_rate = vr
        if vr > ng_void_threshold:
            void_ng_count += 1
            cx, cy = item["solder_circle"]["center"]
            ng_reasons.append(f"焊球 #{idx+1} ({cx:.0f},{cy:.0f}) 气泡率超标: {vr*100:.1f}% > {ng_void_threshold*100:.0f}%")

    board_status = "NG" if void_ng_count > 0 else "PASS"

    summary: Dict[str, Any] = {
        "mode": InspectionMode.VOID.value,
        "board_status": board_status,
        "solder_count": len(void_results),
        "void_pass_count": len(void_results) - void_ng_count,
        "void_ng_count": void_ng_count,
        "max_void_rate": float(max_void_rate),
        "bridge_defect_count": 0,
        "elapsed_ms": round((time.time() - t_start) * 1000, 1),
    }

    out_vis_path = None
    if save_debug_image and vis_img is not None:
        for idx, item in enumerate(void_results):
            cx = int(round(item["solder_circle"]["center"][0]))
            cy = int(round(item["solder_circle"]["center"][1]))
            r = int(round(item["solder_circle"]["radius"]))
            vr = item["void_rate"]
            is_void_ng = vr > ng_void_threshold

            cv2.circle(vis_img, (cx, cy), r, COLOR_SOLDER, 1)
            txt_color = COLOR_TEXT_NG if is_void_ng else COLOR_TEXT_PASS
            cv2.putText(vis_img, f"{vr * 100:.1f}%", (cx - r, max(12, cy - r - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, txt_color, 1, cv2.LINE_AA)

            v_color = COLOR_VOID_NG if is_void_ng else COLOR_VOID_PASS
            for vc in item["void_circle"]:
                vcx, vcy = int(round(vc["center"][0])), int(round(vc["center"][1]))
                vr_px = max(1, int(round(vc["radius"])))
                cv2.circle(vis_img, (vcx, vcy), vr_px, v_color, 1)

        # 绘制顶部 HUD 看板 (气泡专用)
        hud_color = (0, 0, 255) if board_status == "NG" else (0, 200, 0)
        cv2.rectangle(vis_img, (10, 10), (430, 75), (40, 40, 40), -1)
        cv2.rectangle(vis_img, (10, 10), (430, 75), hud_color, 2)
        cv2.putText(vis_img, f"[VOID ONLY] STATUS: {board_status}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, hud_color, 2, cv2.LINE_AA)
        cv2.putText(vis_img, f"Solders: {len(void_results)} | Void-NG: {void_ng_count} | Max-Rate: {max_void_rate*100:.1f}%", 
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        if debug_output_dir is None:
            debug_output_dir = os.path.join(os.path.dirname(__file__), "output", "void_only")
        os.makedirs(debug_output_dir, exist_ok=True)
        base_stem = os.path.splitext(os.path.basename(input_image_path))[0]
        out_vis_path = os.path.join(debug_output_dir, f"{base_stem}_void_inspected.jpg")
        cv2.imencode(".jpg", vis_img)[1].tofile(out_vis_path)

    return {
        "mode": InspectionMode.VOID.value,
        "board_status": board_status,
        "ng_reasons": ng_reasons,
        "summary": summary,
        "void_details": void_results,
        "bridge_details": [],
        "visual_output_path": out_vis_path,
    }


# ==============================================================
# ================= 🌟 [核心质检功能 2: 仅桥连质检] =============
# ==============================================================
def inspect_bga_bridge(
    input_image_path: str,
    weights_path: str,
    conf_threshold: float = 0.50,
    device: str = "0",
    save_debug_image: bool = True,
    debug_output_dir: Optional[str] = None,
) -> BoardInspectionResult:
    """
    🔍 功能二：BGA 焊球【仅桥连短路】质检 (Bridge Only Inspection)
    - 快速定位焊球后直接执行球间桥连短路巡检；
    - 跳过内部气泡分割算法，速度极快（轻量高效）；
    - 整板判定标准：整板存在 0 处桥连则 PASS，发现任何一处桥连直接判 NG。
    """
    t_start = time.time()
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"未找到输入图片: {input_image_path}")

    img = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片数据: {input_image_path}")
    vis_img = img.copy() if save_debug_image else None

    # 1. 快速定位焊点 (YOLO 推理 + 网格自适应精修)
    detector = get_detector_instance(weights_path=weights_path, device=device)
    item = detector.infer_one(input_image_path)
    refined_item = apply_grid_node_refine_to_result_item(item, cfg=GRID_REFINE_CFG)
    data_list = result_to_objects(refined_item, lowercase_keys=True, include_meta=True)

    solder_points = []
    for idx, d in enumerate(data_list):
        conf = float(d.get("conf", 1.0))
        is_grid_added = bool(d.get("added_by_grid", False) or d.get("recovered_by_grid", False))
        if conf_threshold is not None and not is_grid_added and conf < conf_threshold:
            continue
        r = float(d.get("radius", 0))
        if r < 3.5:
            continue
        solder_points.append({
            "CenterX": float(d.get("center_x", 0)),
            "CenterY": float(d.get("center_y", 0)),
            "Radius": r,
            "draw_radius": float(d.get("draw_radius", r) or r),
            "conf": conf,
            "orig_index": idx
        })

    # 2. 桥连缺陷巡检 (紧凑安全 ROI + 实心小锡球测谎 + 双端边缘连通判定)
    bridge_defects: List[BridgeDefect] = detect_bga_bridges(
        image=img,
        solder_points=solder_points,
        result_img=None,
        draw_on_result=False,
    )

    ng_reasons: List[str] = []
    bridge_count = len(bridge_defects)
    for b_idx, bd in enumerate(bridge_defects):
        ng_reasons.append(
            f"焊球对之间存在桥连短路缺陷 #{b_idx+1} [走向: {bd['orientation']}]: "
            f"小锡球中心: {bd['small_ball'][:2]}, 框: {bd['roi_box']}"
        )

    board_status = "NG" if bridge_count > 0 else "PASS"

    summary: Dict[str, Any] = {
        "mode": InspectionMode.BRIDGE.value,
        "board_status": board_status,
        "solder_count": len(solder_points),
        "void_pass_count": len(solder_points),
        "void_ng_count": 0,
        "max_void_rate": 0.0,
        "bridge_defect_count": bridge_count,
        "elapsed_ms": round((time.time() - t_start) * 1000, 1),
    }

    out_vis_path = None
    if save_debug_image and vis_img is not None:
        # A. 绘制焊球外圈
        for p in solder_points:
            cx, cy = int(round(p["CenterX"])), int(round(p["CenterY"]))
            r = int(round(p.get("draw_radius", p["Radius"])))
            cv2.circle(vis_img, (cx, cy), r, COLOR_SOLDER, 1)

        # B. 绘制桥连缺陷 (红色长方形框 + 黄色小锡球 + BRIDGE 标识)
        for bd in bridge_defects:
            rx, ry, rw, rh = bd["roi_box"]
            cv2.rectangle(vis_img, (rx, ry), (rx + rw, ry + rh), COLOR_BRIDGE_BOX, 1)
            bx, by, br = bd["small_ball"]
            cv2.circle(vis_img, (bx, by), br, COLOR_BRIDGE_BALL, 1)
            cv2.putText(vis_img, "BRIDGE", (rx, max(12, ry - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_BRIDGE_BOX, 1, cv2.LINE_AA)

        # C. 绘制顶部 HUD 看板 (桥连专用)
        hud_color = (0, 0, 255) if board_status == "NG" else (0, 200, 0)
        cv2.rectangle(vis_img, (10, 10), (430, 75), (40, 40, 40), -1)
        cv2.rectangle(vis_img, (10, 10), (430, 75), hud_color, 2)
        cv2.putText(vis_img, f"[BRIDGE ONLY] STATUS: {board_status}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, hud_color, 2, cv2.LINE_AA)
        cv2.putText(vis_img, f"Solders: {len(solder_points)} | Bridge Defects: {bridge_count}", 
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        if debug_output_dir is None:
            debug_output_dir = os.path.join(os.path.dirname(__file__), "output", "bridge_only")
        os.makedirs(debug_output_dir, exist_ok=True)
        base_stem = os.path.splitext(os.path.basename(input_image_path))[0]
        out_vis_path = os.path.join(debug_output_dir, f"{base_stem}_bridge_inspected.jpg")
        cv2.imencode(".jpg", vis_img)[1].tofile(out_vis_path)

    return {
        "mode": InspectionMode.BRIDGE.value,
        "board_status": board_status,
        "ng_reasons": ng_reasons,
        "summary": summary,
        "void_details": [],
        "bridge_details": bridge_defects,
        "visual_output_path": out_vis_path,
    }


# ==============================================================
# ================= 🌟 [核心质检功能 3: 全项综合质检] ===========
# ==============================================================
def inspect_bga_comprehensive(
    input_image_path: str,
    weights_path: str,
    conf_threshold: float = 0.50,
    ng_void_threshold: float = 0.25,     # 气泡超标门槛 (默认 25%)
    device: str = "0",
    save_debug_image: bool = True,
    debug_output_dir: Optional[str] = None,
) -> BoardInspectionResult:
    """
    🔍 功能三：BGA 全项【气泡 + 桥连】综合质检 (Comprehensive Inspection)
    - 一站式同步执行焊球定位、气泡空洞率分割与球间桥连检测；
    - 算力高效复用：YOLO 目标检测仅推理 1 次；
    - 整板双重判定：气泡率超标 OR 存在桥连短路缺陷，整板直接判定为 NG；
    - 多图层全景可视化：绿圈焊球 + 红/黄气泡圈 + 红框桥连 + 黄圈小锡球 + 工业 HUD 看板。
    """
    t_start = time.time()
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"未找到输入图片: {input_image_path}")

    img = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片数据: {input_image_path}")
    vis_img = img.copy() if save_debug_image else None

    # 1. 气泡与焊球检测
    void_results: List[PredictionResult] = predict_and_generate_mask(
        model=weights_path,
        input_image_path=input_image_path,
        conf_threshold=conf_threshold,
        ng_threshold=ng_void_threshold * 100.0,
        device=device,
        save_debug_image=False,
    )

    # 2. 桥连检测
    solder_points = []
    for idx, v_item in enumerate(void_results):
        sc = v_item["solder_circle"]
        solder_points.append({
            "CenterX": sc["center"][0],
            "CenterY": sc["center"][1],
            "Radius": sc["radius"],
            "draw_radius": sc["radius"],
            "conf": v_item.get("confidence", 1.0),
            "orig_index": idx
        })

    bridge_defects: List[BridgeDefect] = detect_bga_bridges(
        image=img,
        solder_points=solder_points,
        result_img=None,
        draw_on_result=False,
    )

    # 3. 综合判定
    ng_reasons: List[str] = []
    void_ng_count = 0
    max_void_rate = 0.0
    for idx, item in enumerate(void_results):
        vr = item["void_rate"]
        if vr > max_void_rate:
            max_void_rate = vr
        if vr > ng_void_threshold:
            void_ng_count += 1
            cx, cy = item["solder_circle"]["center"]
            ng_reasons.append(f"焊球 #{idx+1} ({cx:.0f},{cy:.0f}) 气泡率超标: {vr*100:.1f}% > {ng_void_threshold*100:.0f}%")

    bridge_count = len(bridge_defects)
    for b_idx, bd in enumerate(bridge_defects):
        ng_reasons.append(
            f"焊球对之间存在桥连短路缺陷 #{b_idx+1} [走向: {bd['orientation']}]: "
            f"小锡球中心: {bd['small_ball'][:2]}, 框: {bd['roi_box']}"
        )

    board_status = "NG" if (void_ng_count > 0 or bridge_count > 0) else "PASS"

    summary: Dict[str, Any] = {
        "mode": InspectionMode.COMPREHENSIVE.value,
        "board_status": board_status,
        "solder_count": len(void_results),
        "void_pass_count": len(void_results) - void_ng_count,
        "void_ng_count": void_ng_count,
        "max_void_rate": float(max_void_rate),
        "bridge_defect_count": bridge_count,
        "elapsed_ms": round((time.time() - t_start) * 1000, 1),
    }

    # 4. 可视化合成
    out_vis_path = None
    if save_debug_image and vis_img is not None:
        for idx, item in enumerate(void_results):
            cx = int(round(item["solder_circle"]["center"][0]))
            cy = int(round(item["solder_circle"]["center"][1]))
            r = int(round(item["solder_circle"]["radius"]))
            vr = item["void_rate"]
            is_void_ng = vr > ng_void_threshold

            cv2.circle(vis_img, (cx, cy), r, COLOR_SOLDER, 1)
            txt_color = COLOR_TEXT_NG if is_void_ng else COLOR_TEXT_PASS
            cv2.putText(vis_img, f"{vr * 100:.1f}%", (cx - r, max(12, cy - r - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, txt_color, 1, cv2.LINE_AA)

            v_color = COLOR_VOID_NG if is_void_ng else COLOR_VOID_PASS
            for vc in item["void_circle"]:
                vcx, vcy = int(round(vc["center"][0])), int(round(vc["center"][1]))
                vr_px = max(1, int(round(vc["radius"])))
                cv2.circle(vis_img, (vcx, vcy), vr_px, v_color, 1)

        for bd in bridge_defects:
            rx, ry, rw, rh = bd["roi_box"]
            cv2.rectangle(vis_img, (rx, ry), (rx + rw, ry + rh), COLOR_BRIDGE_BOX, 1)
            bx, by, br = bd["small_ball"]
            cv2.circle(vis_img, (bx, by), br, COLOR_BRIDGE_BALL, 1)
            cv2.putText(vis_img, "BRIDGE", (rx, max(12, ry - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_BRIDGE_BOX, 1, cv2.LINE_AA)

        hud_color = (0, 0, 255) if board_status == "NG" else (0, 200, 0)
        cv2.rectangle(vis_img, (10, 10), (450, 75), (40, 40, 40), -1)
        cv2.rectangle(vis_img, (10, 10), (450, 75), hud_color, 2)
        cv2.putText(vis_img, f"[COMPREHENSIVE] STATUS: {board_status}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, hud_color, 2, cv2.LINE_AA)
        cv2.putText(vis_img, f"Solders: {len(void_results)} | Void-NG: {void_ng_count} | Bridges: {bridge_count}", 
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        if debug_output_dir is None:
            debug_output_dir = os.path.join(os.path.dirname(__file__), "output", "comprehensive")
        os.makedirs(debug_output_dir, exist_ok=True)
        base_stem = os.path.splitext(os.path.basename(input_image_path))[0]
        out_vis_path = os.path.join(debug_output_dir, f"{base_stem}_all_inspected.jpg")
        cv2.imencode(".jpg", vis_img)[1].tofile(out_vis_path)

    return {
        "mode": InspectionMode.COMPREHENSIVE.value,
        "board_status": board_status,
        "ng_reasons": ng_reasons,
        "summary": summary,
        "void_details": void_results,
        "bridge_details": bridge_defects,
        "visual_output_path": out_vis_path,
    }


# ==============================================================
# ================= 🌟 [统一总入口函数] =========================
# ==============================================================
def inspect_bga(
    input_image_path: str,
    weights_path: str,
    mode: str = "comprehensive",          # 可选: "comprehensive"(综合) | "void"(仅气泡) | "bridge"(仅桥连)
    conf_threshold: float = 0.50,
    ng_void_threshold: float = 0.25,
    device: str = "0",
    save_debug_image: bool = True,
    debug_output_dir: Optional[str] = None,
) -> BoardInspectionResult:
    """
    🌟 BGA 质检统一总入口 (支持 mode 切换):
    - mode="comprehensive" (默认): 全项综合质检 (气泡 + 桥连)
    - mode="void": 仅气泡质检
    - mode="bridge": 仅桥连质检
    """
    clean_mode = str(mode).strip().lower()
    if clean_mode in ("void", "void_only"):
        return inspect_bga_void(
            input_image_path=input_image_path,
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            ng_void_threshold=ng_void_threshold,
            device=device,
            save_debug_image=save_debug_image,
            debug_output_dir=debug_output_dir,
        )
    elif clean_mode in ("bridge", "bridge_only"):
        return inspect_bga_bridge(
            input_image_path=input_image_path,
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            device=device,
            save_debug_image=save_debug_image,
            debug_output_dir=debug_output_dir,
        )
    else: # 默认 comprehensive
        return inspect_bga_comprehensive(
            input_image_path=input_image_path,
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            ng_void_threshold=ng_void_threshold,
            device=device,
            save_debug_image=save_debug_image,
            debug_output_dir=debug_output_dir,
        )


# 向后兼容别名 (保持旧代码调用不报错)
inspect_bga_board = inspect_bga_comprehensive
