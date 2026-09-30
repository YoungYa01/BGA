from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging
from utils.tool_kit import (
    assign_points_to_standard_grid,
    build_standard_grid,
    build_component_mask,
    clamp_box,
    estimate_main_solder_radius_from_raw,
    estimate_pitch,
    estimate_solder_radius,
    filter_points_to_matrix_components,
    fit_solder_prototype,
    grid_add_only_strict,
    grid_complete_matrix,
    imwrite_unicode,
    point_component_overlap,
    prototype_filter_points,
    recover_removed_points,
    rescue_valid_grid_removed_points,
    rectangle_filter_roi,
)
from utils.torch_utils import TracedModel, select_device, time_synchronized


PointDict = Dict[str, Any]
ResultDict = Dict[str, Any]


def _safe_name(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem or "debug"


def _draw_points(
    image,
    points: List[PointDict],
    box_color=(255, 0, 0),
    circle_color=(0, 0, 255),
    cross_color=(0, 255, 0),
    line_thickness: int = 1,
    circle_thickness: int = 2,
    marker_size: int = 10,
    show_scores: bool = False,
    show_ids: bool = False,
    score_key: str = "Conf",
):
    vis = image
    for idx, p in enumerate(points):
        left, top, right, bottom = p["Left"], p["Top"], p["Right"], p["Bottom"]
        cx, cy = p["CenterX"], p["CenterY"]
        r = int(round(p.get("DrawRadius", p.get("Radius", max(1, min(right - left, bottom - top) / 2.0)))))
        cv2.rectangle(vis, (left, top), (right, bottom), box_color, line_thickness)
        cv2.circle(vis, (cx, cy), max(1, r), circle_color, circle_thickness)
        cv2.drawMarker(vis, (cx, cy), cross_color, cv2.MARKER_CROSS, marker_size, 2)
        label_parts = []
        if show_ids:
            label_parts.append(str(idx))
        if show_scores:
            score_val = p.get(score_key)
            if score_val is not None:
                try:
                    label_parts.append(f"{float(score_val):.2f}")
                except Exception:
                    label_parts.append(str(score_val))
        if label_parts:
            cv2.putText(vis, "|".join(label_parts), (left, max(15, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_raw_detection_stage(result_item: ResultDict, show_scores: bool = False, show_ids: bool = False):
    vis = result_item["image"].copy()
    raw_points = result_item.get("raw_points", [])
    _draw_points(vis, raw_points, box_color=(255, 0, 0), circle_color=(0, 0, 255), cross_color=(0, 255, 0), show_scores=show_scores, show_ids=show_ids, score_key="Conf")
    cv2.putText(vis, f"Stage1 RawDet: {len(raw_points)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    radius_hint = result_item.get("raw_radius_hint", 0.0)
    if radius_hint:
        cv2.putText(vis, f"RawMainR: {radius_hint:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_filtered_stage(result_item: ResultDict, show_scores: bool = False, show_ids: bool = False):
    vis = result_item["image"].copy()
    hard_removed = result_item.get("hard_removed_points_pregrid", result_item.get("hard_removed_points", []))
    soft_removed = result_item.get("soft_removed_points_pregrid", [])
    filtered_points = result_item.get("filtered_points_pregrid", [])
    for p in hard_removed:
        rr = p.get("RejectReason", "")
        color = (180, 180, 180)
        if rr.startswith("via"):
            color = (0, 165, 255)
        cv2.rectangle(vis, (p["Left"], p["Top"]), (p["Right"], p["Bottom"]), color, 1)
    for p in soft_removed:
        cv2.rectangle(vis, (p["Left"], p["Top"]), (p["Right"], p["Bottom"]), (0, 255, 255), 1)
        cv2.circle(vis, (p["CenterX"], p["CenterY"]), max(1, int(round(p.get("Radius", 1)))), (0, 255, 255), 1)
    _draw_points(vis, filtered_points, box_color=(255, 0, 0), circle_color=(0, 255, 0), cross_color=(0, 255, 0), show_scores=show_scores, show_ids=show_ids, score_key="Conf")
    cv2.putText(vis, f"Stage2 Filtered: {len(filtered_points)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"HardRemoved: {len(hard_removed)}  SoftRemoved: {len(soft_removed)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_completed_stage(result_item: ResultDict, show_scores: bool = False, show_ids: bool = False):
    return draw_result_item(result_item, show_scores=show_scores, show_ids=show_ids, show_removed=False)


class BGADetector:
    def __init__(
        self,
        weights: str,
        img_size: int = 1280,
        conf_thres: float = 0.22,
        iou_thres: float = 0.20,
        device: str = "",
        augment: bool = False,
        agnostic_nms: bool = False,
        no_trace: bool = True,
        classes=None,
        enable_component_mask: bool = False,
        component_overlap_hard: float = 0.80,
        component_overlap_soft: float = 0.58,
        min_points_for_grid: int = 12,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        debug_show_scores: bool = False,
        debug_show_ids: bool = False,
    ):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_str = device
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.no_trace = no_trace
        self.classes = classes
        self.enable_component_mask = enable_component_mask
        self.component_overlap_hard = component_overlap_hard
        self.component_overlap_soft = component_overlap_soft
        self.min_points_for_grid = min_points_for_grid
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_show_scores = debug_show_scores
        self.debug_show_ids = debug_show_ids

        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != "cpu"

        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(self.img_size, s=self.stride)

        if not self.no_trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
        if self.half:
            self.model.half()
        if self.debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

    def _run_pre_filter(self, im0, raw_points, component_mask, radius_hint: Optional[float], relax: bool = False):
        prelim_kept: List[PointDict] = []
        hard_removed: List[PointDict] = []
        soft_removed: List[PointDict] = []
        prelim_removed: List[PointDict] = []

        hard_thr = 0.92 if relax else self.component_overlap_hard
        soft_thr = 0.72 if relax else self.component_overlap_soft
        keep_score = 0.85 if relax else 1.10
        hard_remove_score = -2.20 if relax else -1.00

        for p0 in raw_points:
            cur = dict(p0)
            overlap = point_component_overlap(component_mask, cur) if self.enable_component_mask else 0.0
            cur["ComponentOverlap"] = round(float(overlap), 4)
            if self.enable_component_mask and overlap >= hard_thr:
                cur.update({"FilterPassed": False, "FilterStage": "hard_removed", "RejectReason": "component_overlap", "FilterScore": -8.0})
                hard_removed.append(cur)
                prelim_removed.append(cur)
                continue

            keep, meta = rectangle_filter_roi(im0, cur, radius_ref=radius_hint, keep_score=keep_score, hard_remove_score=hard_remove_score)
            cur.update(meta)

            if self.enable_component_mask and overlap >= soft_thr and cur.get("FilterStage") != "hard_removed":
                cur["FilterPassed"] = False
                cur["FilterStage"] = "soft_removed"
                cur["RejectReason"] = "component_overlap_soft"
                soft_removed.append(cur)
                prelim_removed.append(cur)
                continue

            if keep and cur.get("FilterStage") == "kept":
                prelim_kept.append(cur)
            elif cur.get("FilterStage") == "hard_removed":
                hard_removed.append(cur)
                prelim_removed.append(cur)
            else:
                soft_removed.append(cur)
                prelim_removed.append(cur)
        return prelim_kept, hard_removed, soft_removed, prelim_removed

    def _save_stage_debug_images(self, result_item: ResultDict):
        if not self.debug:
            return
        out_dir = self.debug_dir or os.path.join(os.path.dirname(result_item["path"]), "debug_outputs")
        os.makedirs(out_dir, exist_ok=True)
        stem = _safe_name(result_item["path"])
        raw_path = os.path.join(out_dir, f"{stem}_01_raw_detection.png")
        filtered_path = os.path.join(out_dir, f"{stem}_02_filtered.png")
        completed_path = os.path.join(out_dir, f"{stem}_03_completed.png")
        imwrite_unicode(raw_path, draw_raw_detection_stage(result_item, self.debug_show_scores, self.debug_show_ids))
        imwrite_unicode(filtered_path, draw_filtered_stage(result_item, self.debug_show_scores, self.debug_show_ids))
        imwrite_unicode(completed_path, draw_completed_stage(result_item, self.debug_show_scores, self.debug_show_ids))
        result_item["debug_images"] = {"raw_detection": raw_path, "filtered": filtered_path, "completed": completed_path}

    def infer_source(self, source: str) -> List[ResultDict]:
        webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        if webcam:
            _ = check_imshow()
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=self.img_size, stride=self.stride)
        else:
            dataset = LoadImages(source, img_size=self.img_size, stride=self.stride)

        all_results: List[ResultDict] = []
        for path, img, im0s, vid_cap in dataset:
            if webcam:
                frames = im0s
                paths = path
            else:
                frames = [im0s]
                paths = [path]

            img_t = torch.from_numpy(img).to(self.device)
            img_t = img_t.half() if self.half else img_t.float()
            img_t /= 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(img_t, augment=self.augment)[0]
            t2 = time_synchronized()
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            for i, det in enumerate(pred):
                im0 = frames[i].copy() if webcam else frames[0].copy()
                p = str(paths[i] if webcam else paths[0])
                component_mask = build_component_mask(im0) if self.enable_component_mask else None

                image_result: ResultDict = {
                    "filename": os.path.basename(p),
                    "path": p,
                    "infer_ms": round((t2 - t1) * 1000, 2),
                    "nms_ms": round((t3 - t2) * 1000, 2),
                    "raw_count": 0,
                    "shape_removed_count": 0,
                    "prototype_removed_count": 0,
                    "recovered_count": 0,
                    "added_by_grid_count": 0,
                    "total_count": 0,
                    "avg_radius": 0.0,
                    "avg_diameter": 0.0,
                    "raw_radius_hint": 0.0,
                    "pitch": None,
                    "prototype": None,
                    "points": [],
                    "raw_points": [],
                    "filtered_points_pregrid": [],
                    "removed_points": [],
                    "hard_removed_points": [],
                    "soft_removed_points": [],
                    "hard_removed_points_pregrid": [],
                    "soft_removed_points_pregrid": [],
                    "prototype_removed_points": [],
                    "recovered_points": [],
                    "added_points": [],
                    "grid_meta": {},
                    "debug_images": {},
                    "image": im0,
                }

                raw_points: List[PointDict] = []
                if len(det):
                    det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det.cpu().tolist():
                        left, top, right, bottom = map(int, xyxy)
                        h_img, w_img = im0.shape[:2]
                        left, top, right, bottom = clamp_box(left, top, right, bottom, w_img, h_img)
                        w = max(1, right - left)
                        h = max(1, bottom - top)
                        cx = int(round((left + right) / 2.0))
                        cy = int(round((top + bottom) / 2.0))
                        radius = int(round(min(w, h) / 2.0))
                        raw_points.append({
                            "Left": left, "Top": top, "Right": right, "Bottom": bottom,
                            "CenterX": cx, "CenterY": cy, "Radius": radius,
                            "Width": w, "Height": h, "Conf": round(float(conf), 4),
                            "ClassId": int(cls), "AddedByGrid": False, "RecoveredByGrid": False,
                        })

                image_result["raw_count"] = len(raw_points)
                image_result["raw_points"] = [dict(x) for x in raw_points]
                radius_hint = estimate_main_solder_radius_from_raw(im0, raw_points)
                image_result["raw_radius_hint"] = round(float(radius_hint), 3)

                prelim_kept, hard_removed, soft_removed, prelim_removed = self._run_pre_filter(im0, raw_points, component_mask, radius_hint=radius_hint, relax=False)
                if raw_points and len(prelim_kept) < max(8, int(0.08 * len(raw_points))):
                    prelim_kept, hard_removed, soft_removed, prelim_removed = self._run_pre_filter(im0, raw_points, None, radius_hint=radius_hint, relax=True)

                prelim_pitch = estimate_pitch(prelim_kept)
                prototype, enriched_points = fit_solder_prototype(prelim_kept, im0, pitch=prelim_pitch, radius_ref=radius_hint or None)
                prelim_grid_ctx = None
                prelim_grid_meta: Dict[str, Any] = {}
                if prelim_pitch is not None and len(enriched_points) >= self.min_points_for_grid:
                    enriched_points, prelim_grid_ctx, prelim_grid_meta = build_standard_grid(
                        enriched_points,
                        prelim_pitch["pitch_x"],
                        prelim_pitch["pitch_y"],
                    )
                proto_kept, proto_soft_removed, proto_hard_removed = prototype_filter_points(enriched_points, prototype)

                soft_removed.extend(proto_soft_removed)
                hard_removed.extend(proto_hard_removed)
                final_points = proto_kept
                recovered: List[PointDict] = []
                added: List[PointDict] = []
                grid_meta: Dict[str, Any] = {}
                pregrid_rescued: List[PointDict] = []

                if prelim_grid_meta:
                    grid_meta.update({f"pregrid_{k}": v for k, v in prelim_grid_meta.items()})
                pregrid_pitch = estimate_pitch(final_points) if final_points else None
                if pregrid_pitch is None:
                    pregrid_pitch = prelim_pitch
                pregrid_grid_ctx = prelim_grid_ctx
                if pregrid_pitch is not None and len(final_points) >= self.min_points_for_grid:
                    final_points, pregrid_grid_ctx, pregrid_grid_meta = build_standard_grid(
                        final_points,
                        pregrid_pitch["pitch_x"],
                        pregrid_pitch["pitch_y"],
                    )
                    grid_meta.update({f"pregrid_refined_{k}": v for k, v in pregrid_grid_meta.items()})
                if pregrid_grid_ctx is not None and pregrid_pitch is not None and soft_removed:
                    final_points, soft_removed, pregrid_rescued = rescue_valid_grid_removed_points(
                        final_points,
                        soft_removed,
                        pregrid_pitch["pitch_x"],
                        pregrid_pitch["pitch_y"],
                        grid_ctx=pregrid_grid_ctx,
                        prototype=prototype,
                        raw_points=raw_points,
                    )
                    if pregrid_grid_ctx is not None:
                        final_points = assign_points_to_standard_grid(final_points, pregrid_grid_ctx, on_grid_tol_ratio=0.36)
                    grid_meta["pregrid_node_rescued"] = len(pregrid_rescued)

                image_result["filtered_points_pregrid"] = [dict(x) for x in final_points]
                image_result["hard_removed_points_pregrid"] = [dict(x) for x in hard_removed]
                image_result["soft_removed_points_pregrid"] = [dict(x) for x in soft_removed]

                pitch = estimate_pitch(final_points)
                radius_stats = estimate_solder_radius(final_points)
                avg_radius = float(radius_stats.get("avg_radius", 0.0))

                if pitch is not None and len(final_points) >= self.min_points_for_grid:
                    grid_ctx = None
                    final_points, grid_ctx, standard_grid_meta = build_standard_grid(final_points, pitch["pitch_x"], pitch["pitch_y"])
                    grid_meta.update(standard_grid_meta)
                    final_points, soft_removed, recovered = recover_removed_points(
                        final_points,
                        soft_removed,
                        im0,
                        pitch["pitch_x"],
                        pitch["pitch_y"],
                        est_r=avg_radius or radius_hint or None,
                        grid_ctx=grid_ctx,
                        raw_points=raw_points,
                    )
                    matrix_meta = {}
                    matrix_recovered = []
                    matrix_added = []
                    final_points, soft_removed, matrix_recovered, matrix_added, matrix_meta = grid_complete_matrix(
                        final_points,
                        soft_removed,
                        im0,
                        pitch["pitch_x"],
                        pitch["pitch_y"],
                        est_r=avg_radius or radius_hint or None,
                        grid_ctx=grid_ctx,
                        raw_points=raw_points,
                    )
                    if matrix_recovered:
                        recovered.extend(matrix_recovered)
                    if matrix_added:
                        added.extend(matrix_added)
                    final_points, strict_added, strict_meta = grid_add_only_strict(
                        final_points,
                        im0,
                        pitch["pitch_x"],
                        pitch["pitch_y"],
                        est_r=avg_radius or radius_hint or None,
                        grid_ctx=grid_ctx,
                        raw_points=raw_points,
                    )
                    if strict_added:
                        added.extend(strict_added)
                    grid_meta.update(matrix_meta)
                    grid_meta.update(strict_meta)
                    if grid_ctx is not None:
                        final_points = assign_points_to_standard_grid(final_points, grid_ctx, on_grid_tol_ratio=0.36)
                    image_result["pitch"] = {"pitch_x": round(float(pitch["pitch_x"]), 3), "pitch_y": round(float(pitch["pitch_y"]), 3)}

                matrix_windows = []
                if pitch is not None and final_points:
                    final_points, matrix_windows, matrix_meta = filter_points_to_matrix_components(
                        final_points,
                        pitch["pitch_x"],
                        pitch["pitch_y"],
                    )
                    grid_meta.update(matrix_meta)
                    if matrix_windows:
                        recovered = [p for p in recovered if any(
                            win["left"] <= float(p["CenterX"]) <= win["right"] and win["top"] <= float(p["CenterY"]) <= win["bottom"]
                            for win in matrix_windows
                        )]
                        added = [p for p in added if any(
                            win["left"] <= float(p["CenterX"]) <= win["right"] and win["top"] <= float(p["CenterY"]) <= win["bottom"]
                            for win in matrix_windows
                        )]
                    image_result["matrix_windows"] = [dict(x) for x in matrix_windows]

                final_radius_stats = estimate_solder_radius(final_points)
                final_avg_radius = float(final_radius_stats.get("avg_radius", avg_radius))
                final_draw_radius = int(round(final_radius_stats.get("draw_radius", final_avg_radius)))
                for p0 in final_points:
                    p0["DrawRadius"] = final_draw_radius

                image_result["avg_radius"] = round(final_avg_radius, 3)
                image_result["avg_diameter"] = round(final_avg_radius * 2.0, 3)
                image_result["shape_removed_count"] = len(prelim_removed)
                image_result["prototype_removed_count"] = len(proto_soft_removed) + len(proto_hard_removed)
                image_result["removed_points"] = hard_removed + soft_removed
                image_result["hard_removed_points"] = hard_removed
                image_result["soft_removed_points"] = soft_removed
                image_result["prototype_removed_points"] = proto_soft_removed + proto_hard_removed
                image_result["pregrid_rescued_points"] = pregrid_rescued
                image_result["pregrid_rescued_count"] = len(pregrid_rescued)
                image_result["recovered_count"] = len(recovered)
                image_result["recovered_points"] = recovered
                image_result["added_points"] = added
                image_result["added_by_grid_count"] = len(added)
                image_result["grid_meta"] = grid_meta
                image_result["prototype"] = prototype
                image_result["points"] = final_points
                image_result["total_count"] = len(final_points)

                self._save_stage_debug_images(image_result)
                all_results.append(image_result)
        return all_results

    def infer_one(self, image_path: str) -> ResultDict:
        results = self.infer_source(image_path)
        if not results:
            raise ValueError(f"No result returned for source: {image_path}")
        return results[0]


def result_to_objects(result_item: ResultDict, lowercase_keys: bool = True, include_meta: bool = False):
    objects = []
    for p in result_item.get("points", []):
        obj = {
            "x" if lowercase_keys else "Left": p["Left"],
            "y" if lowercase_keys else "Top": p["Top"],
            "w" if lowercase_keys else "Width": p["Width"],
            "h" if lowercase_keys else "Height": p["Height"],
            "center_x" if lowercase_keys else "CenterX": p["CenterX"],
            "center_y" if lowercase_keys else "CenterY": p["CenterY"],
            "radius" if lowercase_keys else "Radius": p["Radius"],
            "draw_radius" if lowercase_keys else "DrawRadius": p.get("DrawRadius", p["Radius"]),
            "conf" if lowercase_keys else "Conf": p.get("Conf", 0.0),
            "class_id" if lowercase_keys else "ClassId": p.get("ClassId", -1),
        }
        if include_meta:
            obj["added_by_grid"] = bool(p.get("AddedByGrid", False))
            obj["recovered_by_grid"] = bool(p.get("RecoveredByGrid", False))
            obj["reject_reason"] = p.get("RejectReason", "")
            obj["via_center_score"] = p.get("ViaCenterScore", p.get("ViaCenterScoreF"))
            obj["radius_ratio"] = p.get("RadiusRatio")
            obj["avg_radius_ref"] = result_item.get("avg_radius", 0.0)
        objects.append(obj)
    return objects


def draw_result_item(result_item: ResultDict, show_scores: bool = False, show_ids: bool = False, show_removed: bool = False):
    image = result_item["image"]
    vis = image.copy()
    removed_points = result_item.get("removed_points", [])
    recovered_points = result_item.get("recovered_points", [])
    final_points = result_item.get("points", [])
    added_points = [p for p in final_points if p.get("AddedByGrid", False)]
    base_points = [p for p in final_points if not p.get("AddedByGrid", False) and not p.get("RecoveredByGrid", False)]
    draw_radius = int(round(result_item.get("avg_radius", 0.0))) if result_item.get("avg_radius", 0.0) > 0 else None

    if show_removed:
        for p in removed_points:
            color = (180, 180, 180)
            rr = p.get("RejectReason", "")
            if rr.startswith("via"):
                color = (0, 165, 255)
            cv2.rectangle(vis, (p["Left"], p["Top"]), (p["Right"], p["Bottom"]), color, 1)

    for idx, p in enumerate(base_points):
        left, top, right, bottom = p["Left"], p["Top"], p["Right"], p["Bottom"]
        cx, cy = p["CenterX"], p["CenterY"]
        r = draw_radius if draw_radius is not None else p["Radius"]
        conf = p.get("Conf", 0.0)
        cv2.rectangle(vis, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.circle(vis, (cx, cy), max(1, int(round(r))), (0, 0, 255), 2)
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
        label_parts = []
        if show_ids:
            label_parts.append(str(idx))
        if show_scores:
            label_parts.append(f"{conf:.2f}")
            if p.get("ViaCenterScore") is not None:
                label_parts.append(f"VC:{float(p.get('ViaCenterScore')):.1f}")
            if p.get("RadiusRatio") is not None:
                label_parts.append(f"RR:{float(p.get('RadiusRatio')):.2f}")
        if label_parts:
            cv2.putText(vis, "|".join(label_parts), (left, max(15, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    for p in recovered_points:
        cx, cy = p["CenterX"], p["CenterY"]
        r = draw_radius if draw_radius is not None else p["Radius"]
        cv2.circle(vis, (cx, cy), max(1, int(round(r))), (255, 255, 0), 2)
        cv2.drawMarker(vis, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 10, 2)

    for p in added_points:
        cx, cy = p["CenterX"], p["CenterY"]
        r = draw_radius if draw_radius is not None else p["Radius"]
        cv2.circle(vis, (cx, cy), max(1, int(round(r))), (0, 255, 255), 2)
        cv2.drawMarker(vis, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 10, 2)

    cv2.putText(vis, f"Stage3 Completed: {result_item.get('total_count', len(final_points))}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"AvgR: {result_item.get('avg_radius', 0.0):.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Added: {result_item.get('added_by_grid_count', 0)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Recovered: {result_item.get('recovered_count', 0)}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"ProtoRemoved: {result_item.get('prototype_removed_count', 0)}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2, cv2.LINE_AA)
    if show_removed:
        cv2.putText(vis, f"Removed: {len(result_item.get('removed_points', []))}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2, cv2.LINE_AA)
    return vis


def save_debug_image(result_item: ResultDict, out_path: str, show_scores: bool = False, show_ids: bool = False, show_removed: bool = False):
    vis = draw_result_item(result_item, show_scores=show_scores, show_ids=show_ids, show_removed=show_removed)
    imwrite_unicode(out_path, vis)
    return out_path


def save_result_json(result_item: ResultDict, out_path: str):
    clean = {k: v for k, v in result_item.items() if k != "image"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    return out_path
