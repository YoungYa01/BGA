import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


PointDict = Dict[str, Any]


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image for saving: {path}")
    buf.tofile(path)


def clamp_box(left: int, top: int, right: int, bottom: int, w: int, h: int):
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    return left, top, right, bottom


def contour_features(cnt):
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    circ = float(4.0 * np.pi * area / (peri * peri + 1e-6))
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = float(max(w, h) / (min(w, h) + 1e-6))
    extent = float(area / (w * h + 1e-6))

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        cx = x + w / 2.0
        cy = y + h / 2.0
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    pts = cnt[:, 0, :].astype(np.float32)
    rr = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    radial_cv = float(np.std(rr) / (np.mean(rr) + 1e-6))

    return {
        "area": area,
        "circularity": circ,
        "aspect": aspect,
        "extent": extent,
        "radial_cv": radial_cv,
        "center_local": (float(cx), float(cy)),
        "bbox_local": (int(x), int(y), int(w), int(h)),
    }


def choose_main_contour(binary, roi_w, roi_h, min_area_ratio=0.02, max_area_ratio=0.96):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    cx0 = roi_w / 2.0
    cy0 = roi_h / 2.0
    roi_area = float(roi_w * roi_h)
    diag = max(1.0, float(np.hypot(roi_w, roi_h)))

    best = None
    best_feat = None
    best_score = -1e9
    for cnt in cnts:
        feat = contour_features(cnt)
        if feat["area"] < roi_area * min_area_ratio or feat["area"] > roi_area * max_area_ratio:
            continue
        cx, cy = feat["center_local"]
        center_penalty = float(np.hypot(cx - cx0, cy - cy0) / diag)
        score = (
            2.2 * feat["circularity"]
            - 1.1 * abs(feat["extent"] - 0.78)
            - 1.4 * feat["radial_cv"]
            - 0.8 * center_penalty
            - 0.5 * max(0.0, feat["aspect"] - 1.0)
        )
        if score > best_score:
            best_score = score
            best = cnt
            best_feat = feat
    return best, best_feat


def radial_intensity_profile(gray: np.ndarray, cx: float, cy: float, r: float):
    yy, xx = np.indices(gray.shape)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    core = gray[dist <= 0.28 * r]
    fill = gray[dist <= 0.72 * r]
    ring = gray[(dist > 0.78 * r) & (dist <= 1.08 * r)]
    outer = gray[(dist > 1.15 * r) & (dist <= 1.65 * r)]

    if core.size < 8 or fill.size < 15 or ring.size < 8 or outer.size < 8:
        return None

    core_m = float(np.mean(core))
    fill_m = float(np.mean(fill))
    ring_m = float(np.mean(ring))
    outer_m = float(np.mean(outer))

    fill_dark_ratio = float(np.mean(fill <= (outer_m - 2.5)))
    core_bright_ratio = float(np.mean(core >= (outer_m + 2.0)))
    ring_dark_ratio = float(np.mean(ring <= (outer_m - 2.0)))

    via_center_score = float(core_m - ring_m)
    via_outer_score = float(outer_m - ring_m)
    solder_fill_dark = float(outer_m - fill_m)
    solder_core_dark = float(outer_m - core_m)

    return {
        "core_mean": core_m,
        "fill_mean": fill_m,
        "ring_mean": ring_m,
        "outer_mean": outer_m,
        "fill_dark_ratio": fill_dark_ratio,
        "core_bright_ratio": core_bright_ratio,
        "ring_dark_ratio": ring_dark_ratio,
        "via_center_score": via_center_score,
        "via_outer_score": via_outer_score,
        "solder_fill_dark": solder_fill_dark,
        "solder_core_dark": solder_core_dark,
    }


def classify_circular_polarity(gray: np.ndarray, cx: float, cy: float, r: float, radius_ref: Optional[float] = None):
    sig = radial_intensity_profile(gray, cx, cy, r)
    if sig is None:
        return {"label": "unknown"}

    rr = float(r / max(1e-6, radius_ref)) if radius_ref else 1.0
    via_center = sig["via_center_score"]
    via_outer = sig["via_outer_score"]
    fill_dark_ratio = sig["fill_dark_ratio"]
    core_bright_ratio = sig["core_bright_ratio"]
    ring_dark_ratio = sig["ring_dark_ratio"]
    solder_fill_dark = sig["solder_fill_dark"]
    solder_core_dark = sig["solder_core_dark"]

    strong_via = (
        rr <= 0.82
        and via_center >= 8.0
        and via_outer >= 3.5
        and (core_bright_ratio >= 0.30 or ring_dark_ratio >= 0.35)
    )
    strong_via_intensity = (
        via_center >= 10.0
        and via_outer >= 5.0
        and ring_dark_ratio >= 0.40
        and solder_core_dark <= 0
    )
    strong_solder = (
        rr >= 0.86
        and solder_fill_dark >= 3.5
        and solder_core_dark >= 2.5
        and fill_dark_ratio >= 0.40
    )

    gray_via_strict = (via_center >= 2.0 and solder_core_dark <= 8.0)
    lack_dark_core = (solder_core_dark <= 1.5 and solder_fill_dark <= 1.5 and fill_dark_ratio <= 0.25)

    gray_via = (
            via_center >= 4.0
            and ring_dark_ratio >= 0.32
            and solder_core_dark <= 1.8
    )

    if (strong_via or strong_via_intensity or gray_via_strict or lack_dark_core) and not strong_solder:
        label = "via"
    elif strong_solder and not strong_via and not strong_via_intensity:
        label = "solder"
    else:
        label = "unknown"

    return {
        "label": label,
        "RadiusRatio": round(rr, 4),
        "CoreMean": round(sig["core_mean"], 3),
        "FillMean": round(sig["fill_mean"], 3),
        "RingMean": round(sig["ring_mean"], 3),
        "OuterMean": round(sig["outer_mean"], 3),
        "FillDarkRatio": round(fill_dark_ratio, 4),
        "CoreBrightRatio": round(core_bright_ratio, 4),
        "RingDarkRatio": round(ring_dark_ratio, 4),
        "ViaCenterScore": round(via_center, 3),
        "ViaOuterScore": round(via_outer, 3),
        "SolderFillDark": round(solder_fill_dark, 3),
        "SolderCoreDark": round(solder_core_dark, 3),
    }


def build_component_mask(
    image_bgr: np.ndarray,
    blur_ksize: int = 11,
    min_area: int = 1200,
    min_side: int = 28,
    dilate_ksize: int = 7,
    max_mask_ratio: float = 0.18,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    th_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    th_dark = cv2.morphologyEx(th_dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    th_dark = cv2.morphologyEx(th_dark, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th_dark, 8)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or max(w, h) < min_side:
            continue
        comp = labels[y:y + h, x:x + w] == i
        if comp.sum() == 0:
            continue
        fill_ratio = float(area / max(1.0, w * h))
        std_val = float(np.std(gray[y:y + h, x:x + w][comp]))
        if fill_ratio >= 0.60 and std_val <= 24.0:
            mask[labels == i] = 255
        elif area >= 10000 and fill_ratio >= 0.42 and std_val <= 28.0:
            mask[labels == i] = 255

    if dilate_ksize > 1 and np.any(mask):
        mask = cv2.dilate(mask, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=1)

    if float(np.mean(mask > 0)) > max_mask_ratio:
        mask[:] = 0
    return mask


def point_component_overlap(mask: np.ndarray, point: PointDict, expand_ratio: float = 0.22) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    H, W = mask.shape[:2]
    left, top, right, bottom = point["Left"], point["Top"], point["Right"], point["Bottom"]
    bw = max(1, right - left)
    bh = max(1, bottom - top)
    pad_x = int(round(bw * expand_ratio))
    pad_y = int(round(bh * expand_ratio))
    l2, t2, r2, b2 = clamp_box(left - pad_x, top - pad_y, right + pad_x, bottom + pad_y, W, H)
    roi = mask[t2:b2, l2:r2]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi > 0))


def _analyze_point_roi(image_bgr: np.ndarray, point: PointDict, roi_pad: float = 0.16, radius_ref: Optional[float] = None):
    H, W = image_bgr.shape[:2]
    left, top, right, bottom = point["Left"], point["Top"], point["Right"], point["Bottom"]
    bw, bh = max(1, right - left), max(1, bottom - top)
    raw_aspect = max(bw, bh) / (min(bw, bh) + 1e-6)

    pad_x = int(round(bw * roi_pad))
    pad_y = int(round(bh * roi_pad))
    l2, t2, r2, b2 = clamp_box(left - pad_x, top - pad_y, right + pad_x, bottom + pad_y, W, H)
    roi = image_bgr[t2:b2, l2:r2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=1)

    c1, f1 = choose_main_contour(th1, roi.shape[1], roi.shape[0], 0.02, 0.96)
    c2, f2 = choose_main_contour(th2, roi.shape[1], roi.shape[0], 0.02, 0.96)
    candidates = [x for x in [(c1, f1), (c2, f2)] if x[0] is not None]
    if not candidates:
        return {
            "raw_aspect": raw_aspect,
            "gray": gray,
            "feat": None,
            "pol": {"label": "unknown"},
            "est_r": float(point.get("Radius", max(1.0, min(bw, bh) / 2.0))),
        }

    feat = max(
        [f for _, f in candidates],
        key=lambda z: (
            2.2 * z["circularity"]
            - 1.1 * abs(z["extent"] - 0.78)
            - 1.4 * z["radial_cv"]
            - 0.5 * max(0.0, z["aspect"] - 1.0)
        ),
    )
    cx, cy = feat["center_local"]
    _, _, bw0, bh0 = feat["bbox_local"]
    est_r = max(4.0, 0.5 * min(bw0, bh0))
    pol = classify_circular_polarity(gray, cx, cy, est_r, radius_ref=radius_ref)
    return {
        "raw_aspect": raw_aspect,
        "gray": gray,
        "feat": feat,
        "pol": pol,
        "est_r": est_r,
    }


def estimate_main_solder_radius_from_raw(image_bgr: np.ndarray, points: List[PointDict]) -> float:
    if not points:
        return 0.0
    enriched = []
    for p in points:
        ana = _analyze_point_roi(image_bgr, p, radius_ref=None)
        if ana is None:
            continue
        pol = ana["pol"]
        feat = ana["feat"]
        rec = dict(p)
        rec.update(pol)
        rec["RadiusF"] = float(ana["est_r"])
        if feat is not None:
            rec["CircularityF"] = float(feat["circularity"])
            rec["RadialCVF"] = float(feat["radial_cv"])
        enriched.append(rec)

    if not enriched:
        radii = np.array([p["Radius"] for p in points], dtype=np.float32)
        return float(np.median(radii)) if radii.size else 0.0

    candidate = []
    for p in enriched:
        if p.get("label") == "via":
            continue
        if p.get("FillDarkRatio", 0.0) >= 0.38 or p.get("SolderFillDark", 0.0) >= 3.0:
            candidate.append(float(p["RadiusF"]))
    if len(candidate) < 8:
        candidate = [float(p["RadiusF"]) for p in enriched]

    vals = np.array(candidate, dtype=np.float32)
    if vals.size == 0:
        return 0.0
    vals = np.sort(vals)
    hi = vals[max(0, int(0.45 * len(vals))):]
    use = hi if hi.size >= 8 else vals
    return float(np.median(use))


def rectangle_filter_roi(
    image_bgr: np.ndarray,
    point: dict,
    radius_ref: Optional[float] = None,
    roi_pad: float = 0.16,
    max_box_aspect: float = 1.85,
    min_circularity: float = 0.40,
    max_extent: float = 0.90,
    max_radial_cv: float = 0.34,
    hard_remove_score: float = -1.0,
    keep_score: float = 1.1,
):
    ana = _analyze_point_roi(image_bgr, point, roi_pad=roi_pad, radius_ref=radius_ref)
    if ana is None:
        return False, {"FilterPassed": False, "RejectReason": "empty_roi", "FilterStage": "hard_removed", "FilterScore": -9.0}

    raw_aspect = ana["raw_aspect"]
    if raw_aspect > max_box_aspect:
        return False, {
            "RejectReason": "bbox_aspect",
            "FilterStage": "hard_removed",
            "FilterScore": -9.0,
            "BoxAspect": round(float(raw_aspect), 4),
            "FilterPassed": False,
            "PolarLabel": "unknown",
        }

    feat = ana["feat"]
    pol = ana["pol"]
    est_r = float(ana["est_r"])
    radius_ratio = float(est_r / max(1e-6, radius_ref)) if radius_ref else 1.0

    if feat is None:
        return False, {
            "FilterPassed": False,
            "RejectReason": "no_contour",
            "FilterStage": "soft_removed",
            "FilterScore": -0.5,
            "PolarLabel": pol.get("label", "unknown"),
            "RadiusRatio": round(radius_ratio, 4),
        }

    via_center = float(pol.get("ViaCenterScore", 0.0))
    via_outer = float(pol.get("ViaOuterScore", 0.0))
    fill_dark = float(pol.get("FillDarkRatio", 0.0))
    solder_fill_dark = float(pol.get("SolderFillDark", 0.0))
    solder_core_dark = float(pol.get("SolderCoreDark", 0.0))
    core_bright_ratio = float(pol.get("CoreBrightRatio", 0.0))
    ring_dark_ratio = float(pol.get("RingDarkRatio", 0.0))
    polar_label = pol.get("label", "unknown")
    conf = float(point.get("Conf", 0.0))

    meta = {
        "BoxAspect": round(float(raw_aspect), 4),
        "ShapeAspect": round(float(feat["aspect"]), 4),
        "Circularity": round(float(feat["circularity"]), 4),
        "Extent": round(float(feat["extent"]), 4),
        "RadialCV": round(float(feat["radial_cv"]), 4),
        "RadiusF": round(est_r, 4),
        "RadiusRatio": round(radius_ratio, 4),
        "PolarLabel": polar_label,
    }
    meta.update(pol)

    # Size-first hard intercept for vias: smaller + bright center + dark ring.
    if radius_ref and radius_ratio <= 0.82 and via_center >= 8.0 and via_outer >= 3.5 and (core_bright_ratio >= 0.30 or ring_dark_ratio >= 0.35):
        meta.update({"FilterPassed": False, "RejectReason": "via_small_centerbright", "FilterStage": "hard_removed", "FilterScore": -5.0})
        return False, meta
    if radius_ref and radius_ratio <= 0.76 and via_center >= 6.0 and fill_dark <= 0.35:
        meta.update({"FilterPassed": False, "RejectReason": "via_small_fillweak", "FilterStage": "hard_removed", "FilterScore": -4.0})
        return False, meta
    if polar_label == "via" and (not radius_ref or radius_ratio <= 0.90):
        meta.update({"FilterPassed": False, "RejectReason": "via_polarity", "FilterStage": "hard_removed", "FilterScore": -4.0})
        return False, meta

    # Size-independent via: strong intensity pattern (bright center + dark ring)
    # even when radius is similar to solder. Key: solder_core_dark <= 0 means
    # center is NOT darker than outer (anti-solder, pro-via signal).
    if via_center >= 10.0 and via_outer >= 5.0 and ring_dark_ratio >= 0.40 and solder_core_dark <= 0 and polar_label != "solder":
        meta.update({"FilterPassed": False, "RejectReason": "via_intensity_strong", "FilterStage": "hard_removed", "FilterScore": -4.5})
        return False, meta
    # ==== 新增：如果在ROI阶段发现是“灰心”或“无黑点”，直接杀掉 ====
    if via_center >= 2.0 and solder_core_dark <= 8.0 and polar_label != "solder":
        meta.update({"FilterPassed": False, "RejectReason": "via_gray_center_strict", "FilterStage": "hard_removed",
                     "FilterScore": -5.0})
        return False, meta
    if solder_core_dark <= 1.5 and solder_fill_dark <= 1.5 and fill_dark <= 0.25:
        meta.update({"FilterPassed": False, "RejectReason": "lack_black_dot", "FilterStage": "hard_removed",
                     "FilterScore": -5.0})
        return False, meta
    # ==== 增加：拦截中心发灰的过孔 ====
    if via_center >= 4.0 and ring_dark_ratio >= 0.32 and solder_core_dark <= 1.8 and polar_label != "solder":
        meta.update({"FilterPassed": False, "RejectReason": "via_gray_center", "FilterStage": "hard_removed", "FilterScore": -4.0})
        return False, meta
    # Strong positive evidence for solder: larger + dark filled.
    if (not radius_ref or radius_ratio >= 0.88) and fill_dark >= 0.40 and solder_fill_dark >= 3.5 and solder_core_dark >= 2.5:
        meta.update({"FilterPassed": True, "RejectReason": "", "FilterStage": "kept", "FilterScore": 3.0})
        return True, meta
    if polar_label == "solder" and (not radius_ref or radius_ratio >= 0.84):
        meta.update({"FilterPassed": True, "RejectReason": "", "FilterStage": "kept", "FilterScore": 2.2})
        return True, meta

    score = 0.0
    score += 1.2 * min(float(feat["circularity"]), 1.0)
    score -= 1.0 * max(0.0, float(feat["aspect"]) - 1.0)
    score -= 1.2 * max(0.0, float(feat["radial_cv"]) - 0.20)
    score -= 0.8 * max(0.0, min_circularity - float(feat["circularity"]))
    score -= 0.7 * max(0.0, float(feat["extent"]) - max_extent)
    score += 0.20 * solder_fill_dark
    score += 0.12 * max(0.0, solder_core_dark)   # reward positive (solder-like dark core)
    score += 0.25 * min(0.0, solder_core_dark)   # penalize negative (via-like bright core)
    score += 1.10 * max(0.0, fill_dark - 0.40)
    if radius_ref:
        score += 1.50 * max(0.0, radius_ratio - 0.85)
        score -= 3.00 * max(0.0, 0.82 - radius_ratio)
    score -= 0.18 * max(0.0, via_center - 6.0)
    score -= 0.16 * max(0.0, via_outer - 3.0)
    score -= 0.22 * max(0.0, ring_dark_ratio - 0.35)  # penalize dark ring (via signal)
    score += 1.1 * conf

    meta["FilterScore"] = round(float(score), 4)

    if radius_ref and radius_ratio <= 0.86 and via_center >= 7.0 and fill_dark <= 0.42 and score <= 0.2:
        meta.update({"FilterPassed": False, "RejectReason": "via_size_score", "FilterStage": "hard_removed"})
        return False, meta

    if score >= keep_score:
        meta.update({"FilterPassed": True, "RejectReason": "", "FilterStage": "kept"})
        return True, meta
    if score <= hard_remove_score:
        meta.update({"FilterPassed": False, "RejectReason": "low_filter_score", "FilterStage": "hard_removed"})
        return False, meta

    meta.update({"FilterPassed": False, "RejectReason": "soft_filter_score", "FilterStage": "soft_removed"})
    return False, meta


def enrich_point_features(image_bgr, point, roi_pad=0.16, radius_ref: Optional[float] = None):
    ana = _analyze_point_roi(image_bgr, point, roi_pad=roi_pad, radius_ref=radius_ref)
    out = dict(point)
    if ana is None:
        out.update({
            "RadiusF": float(point.get("Radius", 0.0)),
            "CircularityF": float(point.get("Circularity", 0.0)),
            "ExtentF": float(point.get("Extent", 0.0)),
            "RadialCVF": float(point.get("RadialCV", 0.0)),
            "ViaCenterScoreF": 0.0,
            "ViaOuterScoreF": 0.0,
            "FillDarkRatioF": 0.0,
            "SolderFillDarkF": 0.0,
            "SolderCoreDarkF": 0.0,
        })
        return out
    feat = ana["feat"]
    pol = ana["pol"]
    out.update(pol)
    out["RadiusF"] = float(ana["est_r"])
    out["ViaCenterScoreF"] = float(pol.get("ViaCenterScore", 0.0))
    out["ViaOuterScoreF"] = float(pol.get("ViaOuterScore", 0.0))
    out["FillDarkRatioF"] = float(pol.get("FillDarkRatio", 0.0))
    out["SolderFillDarkF"] = float(pol.get("SolderFillDark", 0.0))
    out["SolderCoreDarkF"] = float(pol.get("SolderCoreDark", 0.0))
    out["CoreBrightRatioF"] = float(pol.get("CoreBrightRatio", 0.0))
    out["RingDarkRatioF"] = float(pol.get("RingDarkRatio", 0.0))
    out["FilterScoreF"] = float(point.get("FilterScore", 0.0))
    if feat is not None:
        out["CircularityF"] = float(feat["circularity"])
        out["ExtentF"] = float(feat["extent"])
        out["RadialCVF"] = float(feat["radial_cv"])
    else:
        out["CircularityF"] = float(point.get("Circularity", 0.0))
        out["ExtentF"] = float(point.get("Extent", 0.0))
        out["RadialCVF"] = float(point.get("RadialCV", 0.0))
    return out


def select_prototype_seed_points(points: List[PointDict], image_bgr: np.ndarray, pitch: Optional[dict] = None, radius_ref: Optional[float] = None):
    enriched = [enrich_point_features(image_bgr, p, radius_ref=radius_ref) for p in points]
    strong = []
    for p in enriched:
        rr = float(p.get("RadiusRatio", 1.0))
        if (
            p.get("PolarLabel", p.get("label", "unknown")) != "via"
            and p.get("CircularityF", 0.0) >= 0.46
            and p.get("RadialCVF", 1.0) <= 0.32
            and p.get("FillDarkRatioF", 0.0) >= 0.38
            and p.get("SolderFillDarkF", 0.0) >= 3.0
            and (radius_ref is None or rr >= 0.86)
        ):
            strong.append(p)

    if pitch is not None and len(strong) >= 12:
        tmp = [dict(p) for p in strong]
        build_grid_neighbors(tmp, pitch["pitch_x"], pitch["pitch_y"], tol_ratio=0.28)
        comp = largest_dense_component(tmp, min_support=2)
        comp_points = [tmp[i] for i in sorted(comp)] if comp else []
        if len(comp_points) >= 12:
            return comp_points, enriched
    if len(strong) >= 12:
        return strong, enriched
    return enriched, enriched


def estimate_pitch(points, k_neighbors=8):
    if len(points) < 12:
        return None
    xs = np.array([p["CenterX"] for p in points], dtype=np.float32)
    ys = np.array([p["CenterY"] for p in points], dtype=np.float32)
    dxs = []
    dys = []
    for i in range(len(points)):
        dx = np.abs(xs - xs[i])
        dy = np.abs(ys - ys[i])
        mask_x = (dy < 10) & (dx > 3)
        vals_x = dx[mask_x]
        if vals_x.size:
            vals_x = np.sort(vals_x)[:k_neighbors]
            dxs.extend(vals_x.tolist())
        mask_y = (dx < 10) & (dy > 3)
        vals_y = dy[mask_y]
        if vals_y.size:
            vals_y = np.sort(vals_y)[:k_neighbors]
            dys.extend(vals_y.tolist())

    def robust_mode(vals):
        if len(vals) < 10:
            return None
        vals = np.array(vals, dtype=np.float32)
        vals = vals[(vals > 4) & (vals < np.percentile(vals, 85))]
        if vals.size < 10:
            return None
        hist, edges = np.histogram(vals, bins=min(40, max(10, int(np.sqrt(vals.size)))))
        idx = int(np.argmax(hist))
        lo, hi = edges[idx], edges[idx + 1]
        in_bin = vals[(vals >= lo) & (vals <= hi)]
        if in_bin.size == 0:
            return None
        return float(np.median(in_bin))

    px = robust_mode(dxs)
    py = robust_mode(dys)
    if px is None and py is None:
        return None
    if px is None:
        px = py
    if py is None:
        py = px
    return {"pitch_x": float(px), "pitch_y": float(py)}


def _point_is_via_like(point: PointDict, radius_ref: Optional[float] = None) -> bool:
    label = str(point.get("PolarLabel", point.get("label", ""))).lower()
    if label == "via":
        return True
    rad = float(point.get("RadiusF", point.get("Radius", 0.0)))
    base_r = float(radius_ref or point.get("RadiusRef", rad or 1.0) or 1.0)
    rr = float(point.get("RadiusRatio", rad / max(1e-6, base_r)))
    via_center = float(point.get("ViaCenterScoreF", point.get("ViaCenterScore", 0.0)))
    via_outer = float(point.get("ViaOuterScoreF", point.get("ViaOuterScore", 0.0)))
    if rr <= 0.82 and via_center >= 8.0 and via_outer >= 3.5:
        return True
    ring_dark = float(point.get("RingDarkRatioF", point.get("RingDarkRatio", 0.0)))
    fill_dark = float(point.get("FillDarkRatioF", point.get("FillDarkRatio", 0.0)))
    solder_fill = float(point.get("SolderFillDarkF", point.get("SolderFillDark", 0.0)))
    solder_core = float(point.get("SolderCoreDarkF", point.get("SolderCoreDark", 0.0)))
    if via_center >= 10.0 and via_outer >= 5.0 and ring_dark >= 0.40 and solder_core <= 0:
        return True
    # ==== 新增：明确告诉网格系统，这些特征是过孔（Via），绝不允许复活 ====
    if via_center >= 2.0 and solder_core <= 8.0:
        return True
    if solder_core <= 1.5 and solder_fill <= 1.5 and float(
            point.get("FillDarkRatioF", point.get("FillDarkRatio", 0.0))) <= 0.25:
        return True
    # ==== 增加：阻止网格系统将边缘灰过孔当做焊点复活 ====
    if via_center >= 4.0 and ring_dark >= 0.32 and solder_core <= 1.8:
        return True
    return False


def _grid_seed_score(point: PointDict) -> float:
    score = 0.0
    score += 2.0 * float(point.get("Conf", 0.0))
    score += 0.35 * float(point.get("CircularityF", point.get("Circularity", 0.0)))
    score += 0.25 * float(point.get("FillDarkRatioF", point.get("FillDarkRatio", 0.0)))
    score += 0.12 * min(4.0, float(point.get("_support", 0)))
    if point.get("InMainCluster", False):
        score += 0.5
    if point.get("InsideMainWindow", False):
        score += 0.2
    if _point_is_via_like(point):
        score -= 10.0
    return float(score)


def _canonicalize_axis_vector(vec: np.ndarray, axis: str) -> Optional[np.ndarray]:
    out = np.asarray(vec, dtype=np.float32).reshape(2)
    nrm = float(np.linalg.norm(out))
    if nrm < 1e-6:
        return None
    if axis == "col" and out[0] < 0:
        out = -out
    if axis == "row" and out[1] < 0:
        out = -out
    nrm = float(np.linalg.norm(out))
    if nrm < 1e-6:
        return None
    return out


def _estimate_grid_axis_vector(points: List[PointDict], pitch_x: float, pitch_y: float, axis: str, tol_ratio: float = 0.28) -> np.ndarray:
    if not points:
        return np.array([pitch_x, 0.0], dtype=np.float32) if axis == "col" else np.array([0.0, pitch_y], dtype=np.float32)

    tmp = [dict(p) for p in points]
    build_grid_neighbors(tmp, pitch_x, pitch_y, tol_ratio=tol_ratio)

    vectors = []
    keys = ("L", "R") if axis == "col" else ("U", "D")
    pitch_ref = float(pitch_x if axis == "col" else pitch_y)
    for i, p in enumerate(tmp):
        for key in keys:
            j = p["_nbr"].get(key)
            if j is None:
                continue
            q = tmp[j]
            vec = np.array([q["CenterX"] - p["CenterX"], q["CenterY"] - p["CenterY"]], dtype=np.float32)
            vec = _canonicalize_axis_vector(vec, axis)
            if vec is None:
                continue
            length = float(np.linalg.norm(vec))
            if 0.65 * pitch_ref <= length <= 1.45 * pitch_ref:
                vectors.append(vec)

    if not vectors:
        return np.array([pitch_x, 0.0], dtype=np.float32) if axis == "col" else np.array([0.0, pitch_y], dtype=np.float32)

    arr = np.stack(vectors, axis=0)
    lengths = np.linalg.norm(arr, axis=1)
    units = arr / np.maximum(lengths[:, None], 1e-6)
    base = np.median(units, axis=0)
    base = _canonicalize_axis_vector(base, axis)
    if base is None:
        return np.array([pitch_x, 0.0], dtype=np.float32) if axis == "col" else np.array([0.0, pitch_y], dtype=np.float32)
    base /= max(1e-6, float(np.linalg.norm(base)))
    keep = np.sum(units * base[None, :], axis=1) >= 0.86
    kept = arr[keep] if np.any(keep) else arr
    kept_lengths = np.linalg.norm(kept, axis=1)
    length_med = float(np.median(kept_lengths)) if kept_lengths.size else pitch_ref
    direction = np.mean(kept / np.maximum(kept_lengths[:, None], 1e-6), axis=0)
    direction = _canonicalize_axis_vector(direction, axis)
    if direction is None:
        return np.array([pitch_x, 0.0], dtype=np.float32) if axis == "col" else np.array([0.0, pitch_y], dtype=np.float32)
    direction /= max(1e-6, float(np.linalg.norm(direction)))
    return (direction * max(1.0, length_med)).astype(np.float32)


def fit_standard_grid(points: List[PointDict], pitch_x: float, pitch_y: float, tol_ratio: float = 0.28):
    if len(points) < 12:
        return None

    seeds = [dict(p) for p in points if not _point_is_via_like(p)]
    if len(seeds) < 12:
        seeds = [dict(p) for p in points]
    if len(seeds) < 12:
        return None

    seeds.sort(key=_grid_seed_score, reverse=True)
    seeds = seeds[: min(len(seeds), 480)]

    supported = [p for p in seeds if int(p.get("_support", 0)) >= 1 or p.get("InMainCluster", False)]
    if len(supported) >= 12:
        seeds = supported

    col_vec = _estimate_grid_axis_vector(seeds, pitch_x, pitch_y, axis="col", tol_ratio=tol_ratio)
    row_vec = _estimate_grid_axis_vector(seeds, pitch_x, pitch_y, axis="row", tol_ratio=tol_ratio)

    basis = np.array([[col_vec[0], row_vec[0]], [col_vec[1], row_vec[1]]], dtype=np.float32)
    det = float(np.linalg.det(basis))
    if abs(det) < 1e-6:
        return None

    inv_basis = np.linalg.inv(basis)
    col_unit = col_vec / max(1e-6, float(np.linalg.norm(col_vec)))
    row_unit = row_vec / max(1e-6, float(np.linalg.norm(row_vec)))
    seed_arr = np.array([[p["CenterX"], p["CenterY"]] for p in seeds], dtype=np.float32)
    scores = seed_arr @ (col_unit + row_unit)
    origin_xy = seed_arr[int(np.argmin(scores))]

    return {
        "origin": (float(origin_xy[0]), float(origin_xy[1])),
        "basis_col": (float(col_vec[0]), float(col_vec[1])),
        "basis_row": (float(row_vec[0]), float(row_vec[1])),
        "pitch_col": float(np.linalg.norm(col_vec)),
        "pitch_row": float(np.linalg.norm(row_vec)),
        "inv_basis": inv_basis,
        "tol_ratio": float(tol_ratio),
    }


def snap_point_to_standard_grid(cx: float, cy: float, grid_ctx: Optional[Dict[str, Any]], on_grid_tol_ratio: float = 0.34):
    if not grid_ctx:
        return {
            "GridCol": None,
            "GridRow": None,
            "GridColF": None,
            "GridRowF": None,
            "GridResidualPx": None,
            "GridResidualNorm": None,
            "GridX": None,
            "GridY": None,
            "OnStandardGrid": False,
        }

    origin = np.array(grid_ctx["origin"], dtype=np.float32)
    basis_col = np.array(grid_ctx["basis_col"], dtype=np.float32)
    basis_row = np.array(grid_ctx["basis_row"], dtype=np.float32)
    inv_basis = np.asarray(grid_ctx["inv_basis"], dtype=np.float32)
    coord = np.array([cx, cy], dtype=np.float32)

    frac = inv_basis @ (coord - origin)
    col_f = float(frac[0])
    row_f = float(frac[1])
    col = int(np.rint(col_f))
    row = int(np.rint(row_f))

    pred = origin + col * basis_col + row * basis_row
    residual_px = float(np.linalg.norm(coord - pred))
    pitch_ref = max(1e-6, min(float(grid_ctx["pitch_col"]), float(grid_ctx["pitch_row"])))
    residual_norm = residual_px / pitch_ref
    frac_col = abs(col_f - col)
    frac_row = abs(row_f - row)
    tol_frac = min(0.42, max(0.28, float(grid_ctx.get("tol_ratio", 0.28)) + 0.10))
    on_grid = residual_norm <= on_grid_tol_ratio and frac_col <= tol_frac and frac_row <= tol_frac

    return {
        "GridCol": col,
        "GridRow": row,
        "GridColF": round(col_f, 4),
        "GridRowF": round(row_f, 4),
        "GridResidualPx": round(residual_px, 4),
        "GridResidualNorm": round(residual_norm, 4),
        "GridX": float(pred[0]),
        "GridY": float(pred[1]),
        "OnStandardGrid": bool(on_grid),
    }


def grid_node_to_image(node: Tuple[int, int], grid_ctx: Optional[Dict[str, Any]]):
    if not grid_ctx:
        return None
    r, c = node
    origin = np.array(grid_ctx["origin"], dtype=np.float32)
    basis_col = np.array(grid_ctx["basis_col"], dtype=np.float32)
    basis_row = np.array(grid_ctx["basis_row"], dtype=np.float32)
    pred = origin + c * basis_col + r * basis_row
    return float(pred[0]), float(pred[1])


def _grid_ctx_is_dense(grid_ctx: Optional[Dict[str, Any]]) -> bool:
    if not grid_ctx:
        return False
    main_count = len(grid_ctx.get("main_nodes", set()))
    valid_count = len(grid_ctx.get("valid_nodes", set()))
    if valid_count <= 0:
        return False
    return main_count >= 300 and (main_count / max(1, valid_count)) >= 0.72


def assign_points_to_standard_grid(points: List[PointDict], grid_ctx: Optional[Dict[str, Any]], on_grid_tol_ratio: float = 0.34):
    out = []
    valid_nodes = grid_ctx.get("valid_nodes", set()) if grid_ctx else set()
    main_nodes = grid_ctx.get("main_nodes", set()) if grid_ctx else set()
    validity_hint = grid_ctx.get("validity_hint") if grid_ctx else None
    dense_grid = _grid_ctx_is_dense(grid_ctx)
    for p in points:
        q = dict(p)
        snap = snap_point_to_standard_grid(float(q["CenterX"]), float(q["CenterY"]), grid_ctx, on_grid_tol_ratio=on_grid_tol_ratio)
        q.update(snap)
        node_key = (q["GridRow"], q["GridCol"]) if q.get("GridRow") is not None and q.get("GridCol") is not None else None
        q["OnValidGridNode"] = bool(q.get("OnStandardGrid", False) and node_key in valid_nodes)
        q["InStandardGridComponent"] = bool(q.get("OnStandardGrid", False) and node_key in main_nodes)
        q["PotentialValidGridNode"] = bool(q.get("OnStandardGrid", False) and _grid_node_is_potential_valid(node_key, validity_hint=validity_hint))
        q["DenseGridMode"] = dense_grid
        out.append(q)
    return out


def _grid_point_priority(point: PointDict) -> float:
    score = 0.0
    score += 2.5 * float(point.get("Conf", 0.0))
    score -= 0.8 * float(point.get("GridResidualNorm", 0.0) or 0.0)
    score += 0.1 * min(4.0, float(point.get("_support", 0)))
    score += 0.08 * float(point.get("FillDarkRatioF", point.get("FillDarkRatio", 0.0)))
    if point.get("AddedByGrid", False):
        score -= 0.2
    if point.get("RecoveredByGrid", False):
        score -= 0.1
    return float(score)


def _collapse_points_to_grid_nodes(points: List[PointDict]):
    occ: Dict[Tuple[int, int], PointDict] = {}
    for p in points:
        if not p.get("OnStandardGrid", False):
            continue
        if _point_is_via_like(p):
            continue
        row = p.get("GridRow")
        col = p.get("GridCol")
        if row is None or col is None:
            continue
        key = (int(row), int(col))
        prev = occ.get(key)
        if prev is None or _grid_point_priority(p) > _grid_point_priority(prev):
            occ[key] = p
    return occ


def _largest_grid_node_component(node_keys):
    if not node_keys:
        return set()
    node_set = set(node_keys)
    visited = set()
    comps = []
    for start in node_set:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            r, c = cur
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in node_set and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return set(comps[0]) if comps else set()


def _infer_valid_grid_nodes(main_nodes):
    if not main_nodes:
        return set(), {}

    row_to_cols: Dict[int, List[int]] = {}
    col_to_rows: Dict[int, List[int]] = {}
    for r, c in main_nodes:
        row_to_cols.setdefault(int(r), []).append(int(c))
        col_to_rows.setdefault(int(c), []).append(int(r))

    rows = sorted(row_to_cols)
    cols = sorted(col_to_rows)
    if not rows or not cols:
        return set(), {}

    robust_row_bounds = {}
    for r in rows:
        neigh = []
        for rr in range(r - 2, r + 3):
            cs = row_to_cols.get(rr)
            if cs and len(cs) >= 3:
                neigh.append((min(cs), max(cs)))
        if not neigh:
            cs = row_to_cols.get(r, [])
            if cs:
                neigh = [(min(cs), max(cs))]
        if not neigh:
            continue
        left = int(round(np.median([x[0] for x in neigh])))
        right = int(round(np.median([x[1] for x in neigh])))
        if r in row_to_cols:
            cs = row_to_cols[r]
            left = min(left, min(cs))
            right = max(right, max(cs))
        robust_row_bounds[r] = (left, right)

    robust_col_bounds = {}
    for c in cols:
        neigh = []
        for cc in range(c - 2, c + 3):
            rs = col_to_rows.get(cc)
            if rs and len(rs) >= 3:
                neigh.append((min(rs), max(rs)))
        if not neigh:
            rs = col_to_rows.get(c, [])
            if rs:
                neigh = [(min(rs), max(rs))]
        if not neigh:
            continue
        top = int(round(np.median([x[0] for x in neigh])))
        bottom = int(round(np.median([x[1] for x in neigh])))
        if c in col_to_rows:
            rs = col_to_rows[c]
            top = min(top, min(rs))
            bottom = max(bottom, max(rs))
        robust_col_bounds[c] = (top, bottom)

    valid_nodes = set()
    for r, (left, right) in robust_row_bounds.items():
        for c in range(left, right + 1):
            top_bottom = robust_col_bounds.get(c)
            if top_bottom is None:
                continue
            top, bottom = top_bottom
            if top <= r <= bottom:
                valid_nodes.add((int(r), int(c)))

    meta = {
        "row_min": int(min(rows)),
        "row_max": int(max(rows)),
        "col_min": int(min(cols)),
        "col_max": int(max(cols)),
        "row_count": int(len(rows)),
        "col_count": int(len(cols)),
    }
    return valid_nodes, meta


def _build_grid_validity_hint(valid_nodes, main_nodes):
    ref_nodes = set(valid_nodes) if valid_nodes else set(main_nodes)
    if not ref_nodes:
        return {
            "ref_nodes": set(),
            "row_bounds": {},
            "col_bounds": {},
            "row_min": 0,
            "row_max": 0,
            "col_min": 0,
            "col_max": 0,
        }

    row_to_cols: Dict[int, List[int]] = {}
    col_to_rows: Dict[int, List[int]] = {}
    for r, c in ref_nodes:
        row_to_cols.setdefault(int(r), []).append(int(c))
        col_to_rows.setdefault(int(c), []).append(int(r))

    row_bounds = {}
    for r in sorted(row_to_cols):
        neigh = []
        for rr in range(r - 1, r + 2):
            cols = row_to_cols.get(rr)
            if cols and len(cols) >= 2:
                neigh.append((min(cols), max(cols)))
        if not neigh and row_to_cols.get(r):
            cols = row_to_cols[r]
            neigh = [(min(cols), max(cols))]
        if neigh:
            row_bounds[int(r)] = (
                int(round(np.median([x[0] for x in neigh]))),
                int(round(np.median([x[1] for x in neigh]))),
            )

    col_bounds = {}
    for c in sorted(col_to_rows):
        neigh = []
        for cc in range(c - 1, c + 2):
            rows = col_to_rows.get(cc)
            if rows and len(rows) >= 2:
                neigh.append((min(rows), max(rows)))
        if not neigh and col_to_rows.get(c):
            rows = col_to_rows[c]
            neigh = [(min(rows), max(rows))]
        if neigh:
            col_bounds[int(c)] = (
                int(round(np.median([x[0] for x in neigh]))),
                int(round(np.median([x[1] for x in neigh]))),
            )

    rows = [int(r) for r in row_to_cols]
    cols = [int(c) for c in col_to_rows]
    return {
        "ref_nodes": ref_nodes,
        "row_bounds": row_bounds,
        "col_bounds": col_bounds,
        "row_min": int(min(rows)),
        "row_max": int(max(rows)),
        "col_min": int(min(cols)),
        "col_max": int(max(cols)),
    }


def _grid_node_is_potential_valid(node, grid_ctx: Optional[Dict[str, Any]] = None, validity_hint: Optional[Dict[str, Any]] = None, max_extend: int = 1):
    if node is None:
        return False
    if validity_hint is None:
        if grid_ctx is None:
            return False
        validity_hint = grid_ctx.get("validity_hint")
        if validity_hint is None:
            validity_hint = _build_grid_validity_hint(
                grid_ctx.get("valid_nodes", set()),
                grid_ctx.get("main_nodes", set()),
            )

    ref_nodes = set(validity_hint.get("ref_nodes", set()))
    if not ref_nodes:
        return False
    if node in ref_nodes:
        return True

    r, c = int(node[0]), int(node[1])
    row_min = int(validity_hint.get("row_min", r))
    row_max = int(validity_hint.get("row_max", r))
    col_min = int(validity_hint.get("col_min", c))
    col_max = int(validity_hint.get("col_max", c))
    if not (row_min - max_extend <= r <= row_max + max_extend and col_min - max_extend <= c <= col_max + max_extend):
        return False

    neigh_row_bounds = []
    for rr in range(r - 1, r + 2):
        bounds = validity_hint.get("row_bounds", {}).get(rr)
        if bounds is not None:
            neigh_row_bounds.append(bounds)
    if not neigh_row_bounds:
        return False
    left = int(round(np.median([x[0] for x in neigh_row_bounds])))
    right = int(round(np.median([x[1] for x in neigh_row_bounds])))
    if not (left - max_extend <= c <= right + max_extend):
        return False

    neigh_col_bounds = []
    for cc in range(c - 1, c + 2):
        bounds = validity_hint.get("col_bounds", {}).get(cc)
        if bounds is not None:
            neigh_col_bounds.append(bounds)
    if not neigh_col_bounds:
        return False
    top = int(round(np.median([x[0] for x in neigh_col_bounds])))
    bottom = int(round(np.median([x[1] for x in neigh_col_bounds])))
    return bool(top - max_extend <= r <= bottom + max_extend)


def _grid_node_neighbor_stats(occupied_nodes, node):
    occ = set(occupied_nodes)
    r, c = node
    axial = 0
    diag = 0
    has_lr = (r, c - 1) in occ and (r, c + 1) in occ
    has_ud = (r - 1, c) in occ and (r + 1, c) in occ
    for rr2, cc2 in ((r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)):
        if (rr2, cc2) in occ:
            axial += 1
    for rr2, cc2 in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
        if (rr2, cc2) in occ:
            diag += 1
    return axial, diag, has_lr, has_ud


def _grid_node_is_boundary(node, valid_nodes):
    r, c = node
    for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if nxt not in valid_nodes:
            return True
    return False


def build_standard_grid(points: List[PointDict], pitch_x: float, pitch_y: float, tol_ratio: float = 0.28):
    grid_ctx = fit_standard_grid(points, pitch_x, pitch_y, tol_ratio=tol_ratio)
    if grid_ctx is None:
        return [dict(p) for p in points], None, {"standard_grid_fitted": False}

    assigned = assign_points_to_standard_grid(points, grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
    occupancy = _collapse_points_to_grid_nodes(assigned)
    main_nodes = _largest_grid_node_component(set(occupancy.keys()))
    if not main_nodes:
        return assigned, None, {"standard_grid_fitted": False}

    valid_nodes, mask_meta = _infer_valid_grid_nodes(main_nodes)
    grid_ctx = dict(grid_ctx)
    grid_ctx["main_nodes"] = main_nodes
    grid_ctx["valid_nodes"] = valid_nodes if valid_nodes else set(main_nodes)
    grid_ctx.update(mask_meta)
    grid_ctx["validity_hint"] = _build_grid_validity_hint(grid_ctx["valid_nodes"], main_nodes)
    assigned = assign_points_to_standard_grid(points, grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))

    on_grid_count = sum(int(p.get("OnStandardGrid", False)) for p in assigned)
    on_valid_count = sum(int(p.get("OnValidGridNode", False)) for p in assigned)
    public_meta = {
        "standard_grid_fitted": True,
        "origin_x": round(float(grid_ctx["origin"][0]), 3),
        "origin_y": round(float(grid_ctx["origin"][1]), 3),
        "basis_col_x": round(float(grid_ctx["basis_col"][0]), 4),
        "basis_col_y": round(float(grid_ctx["basis_col"][1]), 4),
        "basis_row_x": round(float(grid_ctx["basis_row"][0]), 4),
        "basis_row_y": round(float(grid_ctx["basis_row"][1]), 4),
        "pitch_col": round(float(grid_ctx["pitch_col"]), 4),
        "pitch_row": round(float(grid_ctx["pitch_row"]), 4),
        "main_node_count": int(len(main_nodes)),
        "valid_node_count": int(len(grid_ctx["valid_nodes"])),
        "on_grid_count": int(on_grid_count),
        "on_valid_node_count": int(on_valid_count),
    }
    public_meta.update(mask_meta)
    return assigned, grid_ctx, public_meta


def build_grid_neighbors(points, pitch_x, pitch_y, tol_ratio=0.28):
    tol_x = max(4.0, pitch_x * tol_ratio)
    tol_y = max(4.0, pitch_y * tol_ratio)
    for p in points:
        p["_nbr"] = {"L": None, "R": None, "U": None, "D": None}
        p["_support"] = 0
    for i, p in enumerate(points):
        cx, cy = p["CenterX"], p["CenterY"]
        best = {"L": (1e18, None), "R": (1e18, None), "U": (1e18, None), "D": (1e18, None)}
        for j, q in enumerate(points):
            if i == j:
                continue
            dx = q["CenterX"] - cx
            dy = q["CenterY"] - cy
            if abs(dy) <= tol_y and abs(abs(dx) - pitch_x) <= tol_x:
                score = abs(abs(dx) - pitch_x) + 0.5 * abs(dy)
                if dx < 0 and score < best["L"][0]:
                    best["L"] = (score, j)
                elif dx > 0 and score < best["R"][0]:
                    best["R"] = (score, j)
            if abs(dx) <= tol_x and abs(abs(dy) - pitch_y) <= tol_y:
                score = abs(abs(dy) - pitch_y) + 0.5 * abs(dx)
                if dy < 0 and score < best["U"][0]:
                    best["U"] = (score, j)
                elif dy > 0 and score < best["D"][0]:
                    best["D"] = (score, j)
        for k in ["L", "R", "U", "D"]:
            p["_nbr"][k] = best[k][1]
        p["_support"] = sum(v is not None for v in p["_nbr"].values())
    return points


def largest_dense_component(points, min_support=1):
    valid = [i for i, p in enumerate(points) if p.get("_support", 0) >= min_support]
    valid_set = set(valid)
    comps = []
    visited = set()
    for s in valid:
        if s in visited:
            continue
        q = [s]
        visited.add(s)
        comp = []
        while q:
            cur = q.pop()
            comp.append(cur)
            for nb in points[cur]["_nbr"].values():
                if nb is not None and nb in valid_set and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        comps.append(comp)
    if not comps:
        return set()
    comps.sort(key=lambda c: (len(c), sum(points[i]["_support"] for i in c)), reverse=True)
    return set(comps[0])


def list_dense_components(points, min_support=1, min_size=8):
    valid = [i for i, p in enumerate(points) if p.get("_support", 0) >= min_support]
    valid_set = set(valid)
    comps = []
    visited = set()
    for s in valid:
        if s in visited:
            continue
        q = [s]
        visited.add(s)
        comp = []
        while q:
            cur = q.pop()
            comp.append(cur)
            for nb in points[cur]["_nbr"].values():
                if nb is not None and nb in valid_set and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        if len(comp) >= min_size:
            comps.append(set(comp))
    comps.sort(key=lambda c: (len(c), sum(points[i].get("_support", 0) for i in c)), reverse=True)
    return comps


def find_solder_matrix_windows(points, pitch_x, pitch_y, tol_ratio: float = 0.28, min_size: int = 12):
    if not points or len(points) < min_size:
        return [], {"matrix_component_count": 0, "matrix_window_count": 0}

    tmp = [dict(p) for p in points if not _point_is_via_like(p)]
    if len(tmp) < min_size:
        return [], {"matrix_component_count": 0, "matrix_window_count": 0}

    build_grid_neighbors(tmp, pitch_x, pitch_y, tol_ratio=tol_ratio)
    comps = list_dense_components(tmp, min_support=1, min_size=min_size)
    largest = max((len(c) for c in comps), default=0)
    windows = []

    for comp_id, comp in enumerate(comps):
        comp_points = [tmp[i] for i in sorted(comp)]
        if len(comp_points) < min_size:
            continue
        if largest >= 80 and len(comp_points) < max(min_size, int(0.05 * largest)):
            continue

        xs = [float(p["CenterX"]) for p in comp_points]
        ys = [float(p["CenterY"]) for p in comp_points]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        cols_est = max(1, int(round((x_max - x_min) / max(1e-6, pitch_x))) + 1)
        rows_est = max(1, int(round((y_max - y_min) / max(1e-6, pitch_y))) + 1)
        if rows_est < 3 or cols_est < 3:
            continue

        density = float(len(comp_points)) / max(1.0, float(rows_est * cols_est))
        fill_vals = np.array([float(p.get("FillDarkRatioF", p.get("FillDarkRatio", 0.0))) for p in comp_points], dtype=np.float32)
        solder_vals = np.array([float(p.get("SolderCoreDarkF", p.get("SolderCoreDark", 0.0))) for p in comp_points], dtype=np.float32)
        via_like_frac = float(np.mean([
            1.0 if _point_is_via_like(p, radius_ref=float(p.get("RadiusF", p.get("Radius", 0.0)) or 1.0)) else 0.0
            for p in comp_points
        ]))
        fill_med = float(np.median(fill_vals)) if fill_vals.size else 0.0
        solder_med = float(np.median(solder_vals)) if solder_vals.size else 0.0

        if density < 0.18:
            continue
        if fill_med < 0.30 and solder_med < 3.0:
            continue
        if via_like_frac > 0.40:
            continue

        windows.append({
            "id": int(comp_id),
            "left": float(x_min - 0.55 * pitch_x),
            "top": float(y_min - 0.55 * pitch_y),
            "right": float(x_max + 0.55 * pitch_x),
            "bottom": float(y_max + 0.55 * pitch_y),
            "count": int(len(comp_points)),
            "rows_est": int(rows_est),
            "cols_est": int(cols_est),
            "density": round(float(density), 4),
            "fill_dark_med": round(float(fill_med), 4),
            "solder_dark_med": round(float(solder_med), 4),
        })

    return windows, {
        "matrix_component_count": int(len(comps)),
        "matrix_window_count": int(len(windows)),
    }


def filter_points_to_matrix_windows(points, matrix_windows):
    if not points:
        return [], 0
    if not matrix_windows:
        return [dict(p) for p in points], 0

    kept = []
    removed = 0
    for p in points:
        cx = float(p["CenterX"])
        cy = float(p["CenterY"])
        hit = None
        for win in matrix_windows:
            if win["left"] <= cx <= win["right"] and win["top"] <= cy <= win["bottom"]:
                hit = win
                break
        if hit is None:
            removed += 1
            continue
        q = dict(p)
        q["InsideMatrixWindow"] = True
        q["MatrixComponentId"] = int(hit["id"])
        kept.append(q)
    return kept, int(removed)


def filter_points_to_matrix_components(points, pitch_x, pitch_y, tol_ratio: float = 0.28, min_size: int = 12):
    if not points:
        return [], [], {"matrix_component_count": 0, "matrix_window_count": 0, "matrix_window_filtered_out": 0}

    tmp = []
    for idx, p in enumerate(points):
        if _point_is_via_like(p):
            continue
        q = dict(p)
        q["_MatrixOrigIdx"] = int(idx)
        tmp.append(q)
    if len(tmp) < min_size:
        return [dict(p) for p in points], [], {"matrix_component_count": 0, "matrix_window_count": 0, "matrix_window_filtered_out": 0}

    build_grid_neighbors(tmp, pitch_x, pitch_y, tol_ratio=tol_ratio)
    comps = list_dense_components(tmp, min_support=1, min_size=min_size)
    largest = max((len(c) for c in comps), default=0)
    windows = []
    keep_index_to_comp: Dict[int, int] = {}

    for comp_id, comp in enumerate(comps):
        comp_points = [tmp[i] for i in sorted(comp)]
        if len(comp_points) < min_size:
            continue
        if largest >= 80 and len(comp_points) < max(min_size, int(0.05 * largest)):
            continue

        xs = [float(p["CenterX"]) for p in comp_points]
        ys = [float(p["CenterY"]) for p in comp_points]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        cols_est = max(1, int(round((x_max - x_min) / max(1e-6, pitch_x))) + 1)
        rows_est = max(1, int(round((y_max - y_min) / max(1e-6, pitch_y))) + 1)
        if rows_est < 3 or cols_est < 3:
            continue

        density = float(len(comp_points)) / max(1.0, float(rows_est * cols_est))
        fill_vals = np.array([float(p.get("FillDarkRatioF", p.get("FillDarkRatio", 0.0))) for p in comp_points], dtype=np.float32)
        solder_vals = np.array([float(p.get("SolderCoreDarkF", p.get("SolderCoreDark", 0.0))) for p in comp_points], dtype=np.float32)
        via_like_frac = float(np.mean([
            1.0 if _point_is_via_like(p, radius_ref=float(p.get("RadiusF", p.get("Radius", 0.0)) or 1.0)) else 0.0
            for p in comp_points
        ]))
        fill_med = float(np.median(fill_vals)) if fill_vals.size else 0.0
        solder_med = float(np.median(solder_vals)) if solder_vals.size else 0.0

        if density < 0.18:
            continue
        if fill_med < 0.30 and solder_med < 3.0:
            continue
        if via_like_frac > 0.40:
            continue

        window = {
            "id": int(comp_id),
            "left": float(x_min - 0.40 * pitch_x),
            "top": float(y_min - 0.40 * pitch_y),
            "right": float(x_max + 0.40 * pitch_x),
            "bottom": float(y_max + 0.40 * pitch_y),
            "count": int(len(comp_points)),
            "rows_est": int(rows_est),
            "cols_est": int(cols_est),
            "density": round(float(density), 4),
            "fill_dark_med": round(float(fill_med), 4),
            "solder_dark_med": round(float(solder_med), 4),
        }
        windows.append(window)
        for p in comp_points:
            keep_index_to_comp[int(p["_MatrixOrigIdx"])] = int(comp_id)

    if not windows:
        return [dict(p) for p in points], [], {"matrix_component_count": int(len(comps)), "matrix_window_count": 0, "matrix_window_filtered_out": 0}

    kept = []
    removed = 0
    for idx, p in enumerate(points):
        comp_id = keep_index_to_comp.get(int(idx))
        if comp_id is None:
            hit = None
            cx = float(p["CenterX"])
            cy = float(p["CenterY"])
            for win in windows:
                if win["left"] <= cx <= win["right"] and win["top"] <= cy <= win["bottom"]:
                    hit = win
                    break
            if hit is not None and (
                p.get("AddedByGrid", False)
                or p.get("RecoveredByGrid", False)
                or p.get("OnValidGridNode", False)
                or p.get("PotentialValidGridNode", False)
                or int(p.get("_support", 0)) >= 2
            ):
                comp_id = int(hit["id"])
        if comp_id is None:
            removed += 1
            continue
        q = dict(p)
        q["InsideMatrixWindow"] = True
        q["MatrixComponentId"] = int(comp_id)
        kept.append(q)

    return kept, windows, {
        "matrix_component_count": int(len(comps)),
        "matrix_window_count": int(len(windows)),
        "matrix_window_filtered_out": int(removed),
        "matrix_window_kept": int(len(kept)),
    }


def annotate_grid_membership(points: List[PointDict], pitch_x: float, pitch_y: float, tol_ratio: float = 0.28):
    if not points:
        return points, set(), None
    tmp = [dict(p) for p in points]
    build_grid_neighbors(tmp, pitch_x, pitch_y, tol_ratio=tol_ratio)
    comp = largest_dense_component(tmp, min_support=1)
    if not comp:
        for p, t in zip(points, tmp):
            p["_support"] = t.get("_support", 0)
            p["InMainCluster"] = False
            p["InsideMainWindow"] = False
        return points, set(), None
    main_points = [tmp[i] for i in sorted(comp)]
    x_min = min(p["CenterX"] for p in main_points)
    x_max = max(p["CenterX"] for p in main_points)
    y_min = min(p["CenterY"] for p in main_points)
    y_max = max(p["CenterY"] for p in main_points)
    window = (x_min - 0.35 * pitch_x, y_min - 0.35 * pitch_y, x_max + 0.35 * pitch_x, y_max + 0.35 * pitch_y)
    for idx, (p, t) in enumerate(zip(points, tmp)):
        p["_support"] = t.get("_support", 0)
        p["InMainCluster"] = idx in comp
        p["InsideMainWindow"] = window[0] <= p["CenterX"] <= window[2] and window[1] <= p["CenterY"] <= window[3]
    return points, comp, window


def _grid_assign_indices(points: List[PointDict], pitch_x: float, pitch_y: float):
    if not points:
        return [], (0, 0)
    x0 = min(p["CenterX"] for p in points)
    y0 = min(p["CenterY"] for p in points)
    out = []
    for p in points:
        q = dict(p)
        q["GridCol"] = int(round((p["CenterX"] - x0) / max(1e-6, pitch_x)))
        q["GridRow"] = int(round((p["CenterY"] - y0) / max(1e-6, pitch_y)))
        out.append(q)
    return out, (x0, y0)


def fit_solder_prototype(points, image_bgr, pitch=None, radius_ref: Optional[float] = None):
    if len(points) < 8:
        return None, [enrich_point_features(image_bgr, p, radius_ref=radius_ref) for p in points]
    seeds, enriched = select_prototype_seed_points(points, image_bgr, pitch=pitch, radius_ref=radius_ref)
    if len(seeds) < 8:
        seeds = enriched
    if pitch is not None:
        enriched, _comp, _window = annotate_grid_membership(enriched, pitch["pitch_x"], pitch["pitch_y"], tol_ratio=0.28)

    radii = np.array([p["RadiusF"] for p in seeds], dtype=np.float32)
    big = radii[radii >= np.median(radii) * 0.95] if radii.size else radii
    radii_use = big if big.size >= max(8, int(0.35 * max(1, radii.size))) else radii
    if radius_ref and radii_use.size:
        radii_use = radii_use[radii_use >= 0.82 * radius_ref] if np.any(radii_use >= 0.82 * radius_ref) else radii_use
    darks = np.array([p["SolderCoreDarkF"] for p in seeds], dtype=np.float32)
    via_centers = np.array([p["ViaCenterScoreF"] for p in seeds], dtype=np.float32)
    via_outers = np.array([p["ViaOuterScoreF"] for p in seeds], dtype=np.float32)
    fill_ratios = np.array([p["FillDarkRatioF"] for p in seeds], dtype=np.float32)
    circs = np.array([p["CircularityF"] for p in seeds], dtype=np.float32)
    exts = np.array([p["ExtentF"] for p in seeds], dtype=np.float32)
    rcvs = np.array([p["RadialCVF"] for p in seeds], dtype=np.float32)

    def q(arr, qv):
        return float(np.quantile(arr, qv)) if arr.size else 0.0

    rad_med = float(np.median(radii_use)) if radii_use.size else float(radius_ref or 0.0)
    proto = {
        "radius_med": rad_med,
        "radius_min": max(1.0, q(radii_use, 0.12) * 0.92),
        "radius_max": max(1.0, q(radii_use, 0.90) * 1.18),
        "fill_dark_min": max(0.36, q(fill_ratios, 0.18)),
        "solder_dark_min": q(darks, 0.18),
        "via_center_max": max(7.0, q(via_centers, 0.88)),
        "via_outer_max": max(3.0, q(via_outers, 0.88)),
        "circ_min": max(0.30, q(circs, 0.15)),
        "extent_min": max(0.10, q(exts, 0.10)),
        "extent_max": min(0.98, q(exts, 0.90)),
        "radial_cv_max": max(0.22, q(rcvs, 0.85)),
    }
    return proto, enriched


def _prototype_soft_fail_count(point: PointDict, prototype: Optional[Dict[str, Any]] = None) -> int:
    if prototype is None:
        return 0
    rad = float(point.get("RadiusF", point.get("Radius", 0.0)))
    rr = rad / max(1e-6, float(prototype.get("radius_med", rad if rad > 0 else 1.0)))
    via_center = float(point.get("ViaCenterScoreF", point.get("ViaCenterScore", 0.0)))
    via_outer = float(point.get("ViaOuterScoreF", point.get("ViaOuterScore", 0.0)))
    fill_dark = float(point.get("FillDarkRatioF", point.get("FillDarkRatio", 0.0)))
    solder_dark = float(point.get("SolderCoreDarkF", point.get("SolderCoreDark", 0.0)))
    soft_fail = 0
    if rad < float(prototype.get("radius_min", 0.0)):
        soft_fail += 1
    if rad > float(prototype.get("radius_max", 1e9)) * 1.05:
        soft_fail += 1
    if solder_dark < float(prototype.get("solder_dark_min", solder_dark)) - 1.0:
        soft_fail += 1
    if fill_dark < float(prototype.get("fill_dark_min", fill_dark)) - 0.08:
        soft_fail += 1
    if via_center > float(prototype.get("via_center_max", via_center)) + 1.2 and rr < 0.92:
        soft_fail += 1
    if via_outer > float(prototype.get("via_outer_max", via_outer)) + 1.0 and rr < 0.92:
        soft_fail += 1
    if float(point.get("CircularityF", 1.0)) < float(prototype.get("circ_min", 0.0)) - 0.08:
        soft_fail += 1
    ext = float(point.get("ExtentF", point.get("Extent", 0.8)))
    if ext < float(prototype.get("extent_min", ext)) - 0.10 or ext > float(prototype.get("extent_max", ext)) + 0.10:
        soft_fail += 1
    if float(point.get("RadialCVF", point.get("RadialCV", 0.0))) > float(prototype.get("radial_cv_max", 0.0)) + 0.07:
        soft_fail += 1
    return int(soft_fail)


def _fit_local_solder_prototype(points: List[PointDict], global_proto: Optional[Dict[str, Any]] = None):
    if len(points) < 4:
        return None

    radii = np.array([max(1.0, float(p.get("RadiusF", p.get("Radius", 0.0)))) for p in points], dtype=np.float32)
    darks = np.array([float(p.get("SolderCoreDarkF", p.get("SolderCoreDark", 0.0))) for p in points], dtype=np.float32)
    via_centers = np.array([float(p.get("ViaCenterScoreF", p.get("ViaCenterScore", 0.0))) for p in points], dtype=np.float32)
    via_outers = np.array([float(p.get("ViaOuterScoreF", p.get("ViaOuterScore", 0.0))) for p in points], dtype=np.float32)
    fill_ratios = np.array([float(p.get("FillDarkRatioF", p.get("FillDarkRatio", 0.0))) for p in points], dtype=np.float32)
    circs = np.array([float(p.get("CircularityF", p.get("Circularity", 0.0))) for p in points], dtype=np.float32)
    exts = np.array([float(p.get("ExtentF", p.get("Extent", 0.0))) for p in points], dtype=np.float32)
    rcvs = np.array([float(p.get("RadialCVF", p.get("RadialCV", 0.0))) for p in points], dtype=np.float32)

    def q(arr, qv, default=0.0):
        return float(np.quantile(arr, qv)) if arr.size else float(default)

    proto = {
        "radius_med": float(np.median(radii)),
        "radius_min": max(1.0, q(radii, 0.10) * 0.88),
        "radius_max": max(1.0, q(radii, 0.90) * 1.22),
        "fill_dark_min": max(0.22, q(fill_ratios, 0.15) - 0.03),
        "solder_dark_min": q(darks, 0.15) - 2.0,
        "via_center_max": max(7.0, q(via_centers, 0.90) + 0.8),
        "via_outer_max": max(3.0, q(via_outers, 0.90) + 0.8),
        "circ_min": max(0.24, q(circs, 0.12) - 0.04),
        "extent_min": max(0.05, q(exts, 0.10) - 0.08),
        "extent_max": min(1.02, q(exts, 0.90) + 0.10),
        "radial_cv_max": max(0.24, q(rcvs, 0.88) + 0.05),
    }
    if global_proto is not None:
        proto["via_center_max"] = max(proto["via_center_max"], float(global_proto.get("via_center_max", proto["via_center_max"])))
        proto["via_outer_max"] = max(proto["via_outer_max"], float(global_proto.get("via_outer_max", proto["via_outer_max"])))
        proto["fill_dark_min"] = min(proto["fill_dark_min"], float(global_proto.get("fill_dark_min", proto["fill_dark_min"])))
        proto["circ_min"] = min(proto["circ_min"], float(global_proto.get("circ_min", proto["circ_min"])))
        proto["extent_min"] = min(proto["extent_min"], float(global_proto.get("extent_min", proto["extent_min"])))
        proto["extent_max"] = max(proto["extent_max"], float(global_proto.get("extent_max", proto["extent_max"])))
        proto["radial_cv_max"] = max(proto["radial_cv_max"], float(global_proto.get("radial_cv_max", proto["radial_cv_max"])))
    return proto


def _build_local_prototype_context(points: List[PointDict], prototype: Optional[Dict[str, Any]] = None):
    if not points:
        return {
            "reference_points": [],
            "reference_centers": np.zeros((0, 2), dtype=np.float32),
            "grid_reference_points": {},
        }

    ref_candidates: List[PointDict] = []
    for p in points:
        rad = float(p.get("RadiusF", p.get("Radius", 0.0)))
        fill_dark = float(p.get("FillDarkRatioF", p.get("FillDarkRatio", 0.0)))
        strong_candidate = (
            not _point_is_via_like(p, radius_ref=float(prototype.get("radius_med", max(1.0, rad))) if prototype else None)
            and rad >= max(4.0, 0.55 * float(prototype.get("radius_med", rad if rad > 0 else 4.0)) if prototype else 4.0)
            and (
                _candidate_geometry_ok(p, prototype)
                or int(p.get("_support", 0)) >= 1
                or bool(p.get("InMainCluster", False) or p.get("OnValidGridNode", False))
            )
            and (
                fill_dark >= (float(prototype.get("fill_dark_min", fill_dark)) - 0.16 if prototype else 0.18)
                or int(p.get("_support", 0)) >= 1
                or float(p.get("Conf", 0.0)) >= 0.28
            )
        )
        if strong_candidate:
            ref_candidates.append(p)

    grid_reference_points: Dict[Tuple[int, int], PointDict] = {}
    for p in ref_candidates:
        if not p.get("OnStandardGrid", False):
            continue
        row = p.get("GridRow")
        col = p.get("GridCol")
        if row is None or col is None:
            continue
        key = (int(row), int(col))
        prev = grid_reference_points.get(key)
        if prev is None or _grid_point_priority(p) > _grid_point_priority(prev):
            grid_reference_points[key] = p

    reference_points = list(ref_candidates)
    if reference_points:
        centers = np.array([[float(p["CenterX"]), float(p["CenterY"])] for p in reference_points], dtype=np.float32)
    else:
        centers = np.zeros((0, 2), dtype=np.float32)
    return {
        "reference_points": reference_points,
        "reference_centers": centers,
        "grid_reference_points": grid_reference_points,
    }


def _local_prototype_for_point(point: PointDict, prototype: Optional[Dict[str, Any]], local_ctx: Optional[Dict[str, Any]] = None):
    if local_ctx is None:
        return None

    refs: List[PointDict] = []
    if point.get("OnStandardGrid", False):
        row = point.get("GridRow")
        col = point.get("GridCol")
        if row is not None and col is not None:
            grid_refs = local_ctx.get("grid_reference_points", {})
            for dist in (1, 2, 3):
                for rr in range(int(row) - dist, int(row) + dist + 1):
                    for cc in range(int(col) - dist, int(col) + dist + 1):
                        if rr == int(row) and cc == int(col):
                            continue
                        ref = grid_refs.get((rr, cc))
                        if ref is not None:
                            refs.append(ref)
                if len(refs) >= 6:
                    break

    if len(refs) < 4:
        ref_points = local_ctx.get("reference_points", [])
        centers = local_ctx.get("reference_centers")
        if ref_points and centers is not None and len(ref_points) == len(centers):
            coord = np.array([float(point["CenterX"]), float(point["CenterY"])], dtype=np.float32)
            d2 = np.sum((centers - coord) ** 2, axis=1)
            order = np.argsort(d2)
            max_dist = max(80.0, 6.5 * float(prototype.get("radius_med", point.get("Radius", 12.0))) if prototype else 80.0)
            max_d2 = max_dist * max_dist
            refs = []
            for idx in order.tolist():
                if d2[idx] < 1.0:
                    continue
                if d2[idx] > max_d2 and len(refs) >= 4:
                    break
                refs.append(ref_points[idx])
                if len(refs) >= 12:
                    break

    if len(refs) < 4:
        return None
    return _fit_local_solder_prototype(refs, global_proto=prototype)


def _candidate_geometry_ok(point: PointDict, prototype: Optional[Dict[str, Any]] = None) -> bool:
    circ = float(point.get("CircularityF", point.get("Circularity", 0.0)))
    ext = float(point.get("ExtentF", point.get("Extent", 0.0)))
    radial_cv = float(point.get("RadialCVF", point.get("RadialCV", 1.0)))
    min_circ = 0.34
    min_ext = 0.08
    max_ext = 1.05
    max_radial_cv = 0.45
    if prototype is not None:
        min_circ = max(min_circ, float(prototype.get("circ_min", min_circ)) - 0.16)
        min_ext = max(min_ext, float(prototype.get("extent_min", min_ext)) - 0.20)
        max_ext = min(max_ext, float(prototype.get("extent_max", max_ext)) + 0.20)
        max_radial_cv = max(max_radial_cv, float(prototype.get("radial_cv_max", max_radial_cv)) + 0.15)
    return circ >= min_circ and min_ext <= ext <= max_ext and radial_cv <= max_radial_cv


def _candidate_detection_evidence(point: PointDict, prototype: Optional[Dict[str, Any]] = None) -> bool:
    conf = float(point.get("Conf", 0.0))
    support = int(point.get("_support", 0))
    return (
        (conf >= 0.18 and _candidate_geometry_ok(point, prototype))
        or (support >= 1 and _candidate_geometry_ok(point, prototype))
        or (bool(point.get("InMainCluster", False) or point.get("InStandardGridComponent", False)) and conf >= 0.20)
    )


def prototype_filter_points(points, prototype):
    if not points or prototype is None:
        return points, [], []
    valid_nodes = {
        (int(p["GridRow"]), int(p["GridCol"]))
        for p in points
        if p.get("OnValidGridNode", False) and p.get("GridRow") is not None and p.get("GridCol") is not None
    }
    main_nodes = {
        (int(p["GridRow"]), int(p["GridCol"]))
        for p in points
        if p.get("InStandardGridComponent", False) and p.get("GridRow") is not None and p.get("GridCol") is not None
    }
    validity_hint = _build_grid_validity_hint(valid_nodes, main_nodes)
    local_proto_ctx = _build_local_prototype_context(points, prototype=prototype)
    kept = []
    soft_removed = []
    hard_removed = []
    for p in points:
        cur = dict(p)
        rad = float(cur.get("RadiusF", cur.get("Radius", 0.0)))
        rr = rad / max(1e-6, float(prototype.get("radius_med", rad if rad > 0 else 1.0)))
        via_center = float(cur.get("ViaCenterScoreF", cur.get("ViaCenterScore", 0.0)))
        via_outer = float(cur.get("ViaOuterScoreF", cur.get("ViaOuterScore", 0.0)))
        fill_dark = float(cur.get("FillDarkRatioF", cur.get("FillDarkRatio", 0.0)))
        ring_dark = float(cur.get("RingDarkRatioF", cur.get("RingDarkRatio", 0.0)))
        solder_fill = float(cur.get("SolderFillDarkF", cur.get("SolderFillDark", 0.0)))
        support = int(cur.get("_support", 0))
        on_valid_grid = bool(cur.get("OnValidGridNode", False))
        in_grid_component = bool(cur.get("InStandardGridComponent", False))
        potential_valid_grid = bool(
            cur.get("PotentialValidGridNode", False)
            or (
                cur.get("OnStandardGrid", False)
                and cur.get("GridRow") is not None
                and cur.get("GridCol") is not None
                and _grid_node_is_potential_valid((int(cur["GridRow"]), int(cur["GridCol"])), validity_hint=validity_hint)
            )
        )
        dense_grid_mode = bool(cur.get("DenseGridMode", False))
        local_proto = _local_prototype_for_point(cur, prototype, local_ctx=local_proto_ctx)
        local_soft_fail = _prototype_soft_fail_count(cur, local_proto) if local_proto is not None else None
        if local_soft_fail is not None:
            cur["LocalPrototypeSoftFail"] = int(local_soft_fail)
            cur["LocalPrototypeRadiusMed"] = round(float(local_proto.get("radius_med", 0.0)), 4)

        protected_cluster_point = (
            (support >= 2 or cur.get("InMainCluster", False))
            and rr >= 0.82
            and rr <= 1.20
            and fill_dark >= prototype["fill_dark_min"] - 0.08
        )
        if protected_cluster_point:
            kept.append(cur)
            continue

        grid_protected_point = (
            (on_valid_grid or potential_valid_grid)
            and not _point_is_via_like(cur, radius_ref=float(prototype.get("radius_med", max(1.0, rad))))
            and (
                support >= 2
                or (support >= 1 and _candidate_geometry_ok(cur, prototype))
                or (in_grid_component and _candidate_detection_evidence(cur, prototype))
                or (dense_grid_mode and _candidate_geometry_ok(cur, prototype) and float(cur.get("Conf", 0.0)) >= 0.20)
                or (
                    potential_valid_grid
                    and local_proto is not None
                    and local_soft_fail is not None
                    and local_soft_fail <= 1
                    and float(cur.get("Conf", 0.0)) >= 0.24
                    and _candidate_geometry_ok(cur, local_proto)
                )
            )
        )
        if grid_protected_point:
            cur["GridProtected"] = True
            if potential_valid_grid and not on_valid_grid:
                cur["PotentialValidGridProtected"] = True
            kept.append(cur)
            continue

        if rr <= 0.82 and via_center >= max(8.0, prototype["via_center_max"]) and via_outer >= max(3.5, prototype["via_outer_max"]) and fill_dark <= prototype["fill_dark_min"] - 0.08 and not cur.get("InMainCluster", False):
            cur["RejectReason"] = "via_size_prototype"
            cur["FilterPassed"] = False
            cur["FilterStage"] = "hard_removed"
            hard_removed.append(cur)
            continue

        # Size-independent via: strong intensity pattern regardless of radius ratio
        # solder_core_dark <= 0 means center is NOT darker than outer (anti-solder signal)
        solder_core = float(cur.get("SolderCoreDarkF", cur.get("SolderCoreDark", 0.0)))
        is_strong_via = (via_center >= 10.0 and via_outer >= 5.0 and ring_dark >= 0.40 and solder_core <= 0)
        is_gray_via = (via_center >= 2.0 and solder_core <= 8.0)
        is_lack_dot = (solder_core <= 1.5 and solder_fill <= 1.5 and fill_dark <= 0.25)

        if (is_strong_via or is_gray_via or is_lack_dot) and not cur.get("InMainCluster", False):
            cur["RejectReason"] = "not_black_dot_prototype"
            cur["FilterPassed"] = False
            cur["FilterStage"] = "hard_removed"
            hard_removed.append(cur)
            continue

        soft_fail = _prototype_soft_fail_count(cur, prototype)
        cur["GlobalPrototypeSoftFail"] = int(soft_fail)

        local_proto_rescue = (
            local_proto is not None
            and local_soft_fail is not None
            and local_soft_fail <= 1
            and not _point_is_via_like(cur, radius_ref=float(local_proto.get("radius_med", max(1.0, rad))))
            and (
                potential_valid_grid
                or float(cur.get("Conf", 0.0)) >= 0.30
                or support >= 1
                or bool(cur.get("InMainCluster", False))
            )
        )
        if soft_fail >= 2 and local_proto_rescue:
            cur["LocalPrototypeRescued"] = True
            kept.append(cur)
            continue

        if soft_fail >= 2:
            cur["RejectReason"] = "prototype_mismatch"
            cur["FilterPassed"] = False
            cur["FilterStage"] = "soft_removed"
            soft_removed.append(cur)
        else:
            kept.append(cur)
    return kept, soft_removed, hard_removed


def rescue_valid_grid_removed_points(
    points_kept,
    removed_points,
    pitch_x,
    pitch_y,
    grid_ctx: Optional[Dict[str, Any]] = None,
    prototype: Optional[Dict[str, Any]] = None,
    tol_ratio: float = 0.28,
    center_snap: bool = True,
    raw_points: Optional[List[PointDict]] = None,
):
    if not removed_points:
        return points_kept, removed_points, []

    if grid_ctx is None:
        _, grid_ctx, _ = build_standard_grid(points_kept, pitch_x, pitch_y, tol_ratio=tol_ratio)
    if grid_ctx is None or not grid_ctx.get("valid_nodes"):
        return points_kept, removed_points, []

    valid_nodes = set(grid_ctx["valid_nodes"])
    main_nodes = set(grid_ctx.get("main_nodes", set()))
    validity_hint = grid_ctx.get("validity_hint") or _build_grid_validity_hint(valid_nodes, main_nodes)
    dense_grid_mode = _grid_ctx_is_dense(grid_ctx)
    current = assign_points_to_standard_grid(points_kept, grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
    current = [dict(p) for p in current]
    occupancy = _collapse_points_to_grid_nodes(current)
    est_r = float(prototype.get("radius_med", 0.0)) if prototype else None
    raw_anchor_index = _build_raw_anchor_index(raw_points, grid_ctx, pitch_x, pitch_y, est_r=est_r)

    pending = [dict(p) for p in removed_points]
    rescued = []
    changed = True

    while changed and pending:
        changed = False
        next_pending = []
        for p in pending:
            if _point_is_via_like(p, radius_ref=float(prototype.get("radius_med", p.get("Radius", 1.0))) if prototype else None):
                next_pending.append(p)
                continue

            snap = snap_point_to_standard_grid(float(p["CenterX"]), float(p["CenterY"]), grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
            if not snap["OnStandardGrid"]:
                next_pending.append(p)
                continue

            node_key = (int(snap["GridRow"]), int(snap["GridCol"]))
            node_valid = node_key in valid_nodes
            potential_valid = _grid_node_is_potential_valid(node_key, validity_hint=validity_hint)
            if node_key in occupancy or (not node_valid and not potential_valid):
                next_pending.append(p)
                continue

            boundary_candidate = _grid_node_is_boundary(node_key, valid_nodes | ({node_key} if potential_valid else set()))
            axial, diag, has_lr, has_ud = _grid_node_neighbor_stats(set(occupancy.keys()), node_key)
            if boundary_candidate:
                enough_support = axial >= 2 or has_lr or has_ud or (axial >= 1 and diag >= 2)
            else:
                enough_support = has_lr or has_ud or axial >= 2
            if (not enough_support) and potential_valid and _candidate_detection_evidence(p, prototype):
                enough_support = (axial + diag) >= 1 or float(p.get("Conf", 0.0)) >= 0.38
            if (not enough_support) and dense_grid_mode and _candidate_detection_evidence(p, prototype) and float(p.get("Conf", 0.0)) >= 0.20:
                enough_support = True
            if not enough_support:
                next_pending.append(p)
                continue

            if not _candidate_detection_evidence(p, prototype):
                next_pending.append(p)
                continue

            new_p = dict(p)
            if center_snap:
                cx = int(round(snap["GridX"]))
                cy = int(round(snap["GridY"]))
                est_r = float(prototype.get("radius_med", p.get("Radius", 8.0))) if prototype else float(p.get("Radius", 8.0))
                est_r = max(1.0, est_r)
                anchor_point, anchor_source = _resolve_node_anchor(
                    node_key,
                    grid_ctx,
                    pitch_x,
                    pitch_y,
                    est_r=est_r,
                    preferred_points=[p],
                    raw_anchor_index=raw_anchor_index,
                )
                _apply_anchor_geometry(new_p, anchor_point, cx, cy, est_r, anchor_source=anchor_source)
            new_p.update(snap)
            new_p["RecoveredByGrid"] = True
            new_p["RecoveredByNodeRule"] = True
            new_p["AddedByGrid"] = False
            new_p["OnValidGridNode"] = bool(node_valid)
            new_p["PotentialValidGridNode"] = bool(potential_valid)
            new_p["InStandardGridComponent"] = node_key in main_nodes
            if potential_valid and not node_valid:
                new_p["RecoveredByPotentialGridNode"] = True
            current.append(new_p)
            occupancy[node_key] = new_p
            rescued.append(new_p)
            changed = True

        pending = next_pending

    return current, pending, rescued


def estimate_solder_radius(points):
    if not points:
        return {"avg_radius": 0.0, "draw_radius": 0}
    radii = np.array([max(1.0, float(p.get("RadiusF", p.get("Radius", 0.0)))) for p in points], dtype=np.float32)
    avg_r = float(np.median(radii))
    return {"avg_radius": avg_r, "draw_radius": int(round(avg_r))}


def neighbor_support(points, cx, cy, pitch_x, pitch_y, tol_ratio=0.28):
    tol_x = max(4.0, pitch_x * tol_ratio)
    tol_y = max(4.0, pitch_y * tol_ratio)
    hits = {"L": None, "R": None, "U": None, "D": None}
    for p in points:
        dx = p["CenterX"] - cx
        dy = p["CenterY"] - cy
        if abs(dy) <= tol_y and abs(abs(dx) - pitch_x) <= tol_x:
            score = abs(abs(dx) - pitch_x) + 0.5 * abs(dy)
            if dx < 0 and (hits["L"] is None or score < hits["L"][0]):
                hits["L"] = (score, p)
            elif dx > 0 and (hits["R"] is None or score < hits["R"][0]):
                hits["R"] = (score, p)
        if abs(dx) <= tol_x and abs(abs(dy) - pitch_y) <= tol_y:
            score = abs(abs(dy) - pitch_y) + 0.5 * abs(dx)
            if dy < 0 and (hits["U"] is None or score < hits["U"][0]):
                hits["U"] = (score, p)
            elif dy > 0 and (hits["D"] is None or score < hits["D"][0]):
                hits["D"] = (score, p)
    out = {k: (hits[k][1] if hits[k] is not None else None) for k in hits}
    support = sum(v is not None for v in out.values())
    has_lr = out["L"] is not None and out["R"] is not None
    has_ud = out["U"] is not None and out["D"] is not None
    return out, support, has_lr, has_ud


def roi_has_solder_ball(image_bgr, cx, cy, est_r, radius_ref=None):
    H, W = image_bgr.shape[:2]
    r = max(4, int(round(est_r)))
    x1, y1, x2, y2 = clamp_box(cx - 2 * r, cy - 2 * r, cx + 2 * r, cy + 2 * r, W, H)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False, {}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rx = cx - x1
    ry = cy - y1
    pol = classify_circular_polarity(gray, rx, ry, r, radius_ref=radius_ref)
    rr = float(pol.get("RadiusRatio", 1.0))
    via_center = float(pol.get("ViaCenterScore", 0.0))
    via_outer = float(pol.get("ViaOuterScore", 0.0))
    fill_dark = float(pol.get("FillDarkRatio", 0.0))
    solder_fill_dark = float(pol.get("SolderFillDark", 0.0))
    solder_core_dark = float(pol.get("SolderCoreDark", 0.0))
    if rr <= 0.82 and via_center >= 8.0 and via_outer >= 3.5:
        meta = {"LocalType": "via"}
        meta.update(pol)
        return False, meta
    # ==== 新增：边缘网格扫描遇到“灰过孔”或“无黑点区域”直接否决 ====
    if (via_center >= 2.0 and solder_core_dark <= 8.0) or (solder_core_dark <= 1.5 and solder_fill_dark <= 1.5):
        meta = {"LocalType": "via_or_noise"}
        meta.update(pol)
        return False, meta
    if pol.get("label") == "solder":
        meta = {"LocalType": "solder"}
        meta.update(pol)
        return True, meta
    ok = (rr >= 0.86 and fill_dark >= 0.40 and solder_fill_dark >= 3.5 and solder_core_dark >= 2.5)
    meta = {"LocalType": "unknown"}
    meta.update(pol)
    return ok, meta


def _estimate_local_node_radius(occupancy: Dict[Tuple[int, int], PointDict], node_key: Tuple[int, int], default_r: float) -> float:
    r, c = node_key
    vals = []
    for dist in (1, 2):
        for rr in range(r - dist, r + dist + 1):
            for cc in range(c - dist, c + dist + 1):
                if rr == r and cc == c:
                    continue
                p = occupancy.get((rr, cc))
                if p is None:
                    continue
                rad = float(p.get("RadiusF", p.get("Radius", 0.0)))
                if rad > 0:
                    vals.append(rad)
        if len(vals) >= 4:
            break
    if not vals:
        return float(default_r)
    return float(np.median(np.array(vals, dtype=np.float32)))


def scan_empty_grid_node(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    est_r: float,
    radius_ref: Optional[float] = None,
    seed_conf: float = 0.22,
):
    est_r = max(4.0, float(est_r))
    offsets = [(0.0, 0.0)]
    for frac in (0.16, 0.28):
        d = frac * est_r
        offsets.extend([
            (-d, 0.0), (d, 0.0), (0.0, -d), (0.0, d),
        ])
    d = 0.18 * est_r
    offsets.extend([(-d, -d), (-d, d), (d, -d), (d, d)])
    radius_scales = (0.90, 1.00, 1.10)

    best = None
    best_score = -1e18
    for scale in radius_scales:
        cur_r = max(4.0, float(est_r * scale))
        for dx, dy in offsets:
            px = int(round(cx + dx))
            py = int(round(cy + dy))
            cand = {
                "Left": int(round(px - cur_r)),
                "Top": int(round(py - cur_r)),
                "Right": int(round(px + cur_r)),
                "Bottom": int(round(py + cur_r)),
                "CenterX": px,
                "CenterY": py,
                "Radius": int(round(cur_r)),
                "Width": int(round(2 * cur_r)),
                "Height": int(round(2 * cur_r)),
                "Conf": float(seed_conf),
            }
            keep, meta = rectangle_filter_roi(
                image_bgr,
                cand,
                radius_ref=radius_ref or est_r,
                roi_pad=0.20,
                keep_score=0.55,
                hard_remove_score=-2.20,
            )
            if meta.get("RejectReason", "").startswith("via") or _point_is_via_like(meta, radius_ref=radius_ref or est_r):
                continue
            offset_penalty = 0.80 * (float(np.hypot(dx, dy)) / max(1e-6, est_r))
            scale_penalty = 0.45 * abs(scale - 1.0)
            score = float(meta.get("FilterScore", -9.0)) - offset_penalty - scale_penalty
            if keep:
                score += 0.45
            if meta.get("PolarLabel") == "solder":
                score += 0.35
            if score > best_score:
                best_score = score
                best = (cand, dict(meta))

    if best is None:
        return False, {}
    cand, meta = best
    ok = (
        best_score >= 0.45
        and not meta.get("RejectReason", "").startswith("via")
        and meta.get("PolarLabel") != "via"
    )
    if not ok:
        return False, {}
    meta = dict(meta)
    meta["ScanScore"] = round(float(best_score), 4)
    meta["ScanRecovered"] = True
    meta["LocalType"] = meta.get("LocalType", "solder_scan")
    meta["CenterX"] = cand["CenterX"]
    meta["CenterY"] = cand["CenterY"]
    meta["Radius"] = cand["Radius"]
    meta["Width"] = cand["Width"]
    meta["Height"] = cand["Height"]
    meta["Left"] = cand["Left"]
    meta["Top"] = cand["Top"]
    meta["Right"] = cand["Right"]
    meta["Bottom"] = cand["Bottom"]
    return True, meta


def point_exists_near(points, cx, cy, tol):
    for p in points:
        if abs(p["CenterX"] - cx) <= tol and abs(p["CenterY"] - cy) <= tol:
            return True
    return False


def _point_matches_grid_node(point: PointDict, node_key: Tuple[int, int], grid_ctx: Optional[Dict[str, Any]], pitch_x: float, pitch_y: float, est_r: Optional[float] = None, on_grid_tol_ratio: float = 0.48):
    if point is None or grid_ctx is None:
        return False, {}
    snap = snap_point_to_standard_grid(float(point["CenterX"]), float(point["CenterY"]), grid_ctx, on_grid_tol_ratio=on_grid_tol_ratio)
    if snap.get("OnStandardGrid", False):
        key = (int(snap["GridRow"]), int(snap["GridCol"]))
        if key == node_key:
            return True, snap
    center = grid_node_to_image(node_key, grid_ctx)
    if center is None:
        return False, snap
    dx = float(point["CenterX"]) - float(center[0])
    dy = float(point["CenterY"]) - float(center[1])
    dist = float(np.hypot(dx, dy))
    pitch_ref = max(1e-6, min(float(pitch_x), float(pitch_y)))
    rad = max(1.0, float(point.get("RadiusF", point.get("Radius", est_r or 0.0))))
    tol_px = max(0.42 * pitch_ref, 1.8 * max(rad, float(est_r or rad)))
    return dist <= tol_px, snap


def _anchor_candidate_score(point: PointDict, node_key: Tuple[int, int], grid_ctx: Optional[Dict[str, Any]], pitch_x: float, pitch_y: float, est_r: Optional[float] = None, snap: Optional[Dict[str, Any]] = None):
    if point is None:
        return -1e18
    if snap is None:
        _, snap = _point_matches_grid_node(point, node_key, grid_ctx, pitch_x, pitch_y, est_r=est_r)
    center = grid_node_to_image(node_key, grid_ctx)
    if center is not None:
        dist = float(np.hypot(float(point["CenterX"]) - float(center[0]), float(point["CenterY"]) - float(center[1])))
    else:
        dist = 0.0
    pitch_ref = max(1e-6, min(float(pitch_x), float(pitch_y)))
    rad = max(1.0, float(point.get("RadiusF", point.get("Radius", est_r or 1.0))))
    aspect = float(max(float(point.get("Width", 2 * rad)), float(point.get("Height", 2 * rad))) / max(1.0, min(float(point.get("Width", 2 * rad)), float(point.get("Height", 2 * rad)))))
    score = 4.5 * float(point.get("Conf", 0.0))
    score -= 1.8 * (dist / pitch_ref)
    if est_r is not None and est_r > 1e-6:
        score -= 0.55 * abs(rad - float(est_r)) / float(est_r)
    score -= 0.35 * max(0.0, aspect - 1.0)
    score -= 0.25 * float((snap or {}).get("GridResidualNorm", 0.0) or 0.0)
    return float(score)


def _build_raw_anchor_index(raw_points: Optional[List[PointDict]], grid_ctx: Optional[Dict[str, Any]], pitch_x: float, pitch_y: float, est_r: Optional[float] = None):
    anchor_index: Dict[Tuple[int, int], PointDict] = {}
    if not raw_points or grid_ctx is None:
        return anchor_index
    for raw in raw_points:
        aspect = float(max(float(raw.get("Width", 1.0)), float(raw.get("Height", 1.0))) / max(1.0, min(float(raw.get("Width", 1.0)), float(raw.get("Height", 1.0)))))
        if aspect > 1.55:
            continue
        snap = snap_point_to_standard_grid(float(raw["CenterX"]), float(raw["CenterY"]), grid_ctx, on_grid_tol_ratio=0.50)
        if not snap.get("OnStandardGrid", False):
            continue
        key = (int(snap["GridRow"]), int(snap["GridCol"]))
        matched, _ = _point_matches_grid_node(raw, key, grid_ctx, pitch_x, pitch_y, est_r=est_r, on_grid_tol_ratio=0.50)
        if not matched:
            continue
        cand = dict(raw)
        cand.update(snap)
        prev = anchor_index.get(key)
        if prev is None or _anchor_candidate_score(cand, key, grid_ctx, pitch_x, pitch_y, est_r=est_r, snap=snap) > _anchor_candidate_score(prev, key, grid_ctx, pitch_x, pitch_y, est_r=est_r):
            anchor_index[key] = cand
    return anchor_index


def _resolve_node_anchor(node_key: Tuple[int, int], grid_ctx: Optional[Dict[str, Any]], pitch_x: float, pitch_y: float, est_r: Optional[float] = None, preferred_points: Optional[List[PointDict]] = None, raw_anchor_index: Optional[Dict[Tuple[int, int], PointDict]] = None):
    candidates: List[Tuple[float, PointDict, str]] = []
    for point in preferred_points or []:
        matched, snap = _point_matches_grid_node(point, node_key, grid_ctx, pitch_x, pitch_y, est_r=est_r)
        if not matched:
            continue
        cand = dict(point)
        if snap:
            cand.update(snap)
        candidates.append((_anchor_candidate_score(cand, node_key, grid_ctx, pitch_x, pitch_y, est_r=est_r, snap=snap), cand, "preferred"))
    if raw_anchor_index is not None:
        raw = raw_anchor_index.get(node_key)
        if raw is not None:
            candidates.append((_anchor_candidate_score(raw, node_key, grid_ctx, pitch_x, pitch_y, est_r=est_r), dict(raw), "raw"))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    return best[1], best[2]


def _apply_anchor_geometry(out_point: PointDict, anchor_point: Optional[PointDict], fallback_cx: int, fallback_cy: int, fallback_r: float, anchor_source: Optional[str] = None):
    use = anchor_point if anchor_point is not None else {}
    use_r = float(use.get("RadiusF", use.get("Radius", fallback_r)))
    if use_r <= 0:
        use_r = float(fallback_r)
    use_cx = int(round(float(use.get("CenterX", fallback_cx))))
    use_cy = int(round(float(use.get("CenterY", fallback_cy))))
    use_w = int(round(float(use.get("Width", 2 * use_r))))
    use_h = int(round(float(use.get("Height", 2 * use_r))))
    if use_w <= 0:
        use_w = int(round(2 * use_r))
    if use_h <= 0:
        use_h = int(round(2 * use_r))
    out_point["CenterX"] = use_cx
    out_point["CenterY"] = use_cy
    out_point["Radius"] = int(round(use_r))
    out_point["Width"] = int(use_w)
    out_point["Height"] = int(use_h)
    if anchor_point is not None and all(k in use for k in ("Left", "Top", "Right", "Bottom")):
        out_point["Left"] = int(round(float(use["Left"])))
        out_point["Top"] = int(round(float(use["Top"])))
        out_point["Right"] = int(round(float(use["Right"])))
        out_point["Bottom"] = int(round(float(use["Bottom"])))
    else:
        out_point["Left"] = int(round(use_cx - use_w / 2.0))
        out_point["Top"] = int(round(use_cy - use_h / 2.0))
        out_point["Right"] = int(round(use_cx + use_w / 2.0))
        out_point["Bottom"] = int(round(use_cy + use_h / 2.0))
    if anchor_source:
        out_point["GeometryAnchorSource"] = anchor_source
    out_point["UsedDetectionAnchor"] = bool(anchor_point is not None)
    return out_point


def recover_removed_points(points_kept, removed_points, image_bgr, pitch_x, pitch_y, tol_ratio=0.28, est_r=None, grid_ctx: Optional[Dict[str, Any]] = None, raw_points: Optional[List[PointDict]] = None):
    if not removed_points:
        return points_kept, removed_points, []

    if grid_ctx is None:
        _, grid_ctx, _ = build_standard_grid(points_kept, pitch_x, pitch_y, tol_ratio=tol_ratio)
    if grid_ctx is None or not grid_ctx.get("valid_nodes"):
        return points_kept, removed_points, []

    valid_nodes = set(grid_ctx["valid_nodes"])
    dense_grid_mode = _grid_ctx_is_dense(grid_ctx)
    current = assign_points_to_standard_grid(points_kept, grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
    current = [dict(p) for p in current]
    occupancy = _collapse_points_to_grid_nodes(current)

    recovered = []
    still_removed = []
    if est_r is None:
        est_r = float(np.median([p.get("Radius", 8.0) for p in current])) if current else 8.0
    raw_anchor_index = _build_raw_anchor_index(raw_points, grid_ctx, pitch_x, pitch_y, est_r=est_r)

    for p in removed_points:
        if _point_is_via_like(p, radius_ref=est_r):
            still_removed.append(p)
            continue

        snap = snap_point_to_standard_grid(float(p["CenterX"]), float(p["CenterY"]), grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
        if not snap["OnStandardGrid"]:
            still_removed.append(p)
            continue

        node_key = (int(snap["GridRow"]), int(snap["GridCol"]))
        if node_key not in valid_nodes or node_key in occupancy:
            still_removed.append(p)
            continue

        boundary_candidate = _grid_node_is_boundary(node_key, valid_nodes)
        axial, diag, has_lr, has_ud = _grid_node_neighbor_stats(set(occupancy.keys()), node_key)
        if boundary_candidate:
            enough_support = axial >= 2 or has_lr or has_ud or (axial >= 1 and diag >= 2)
        else:
            enough_support = has_lr or has_ud or axial >= 2
        if (not enough_support) and dense_grid_mode and _candidate_detection_evidence(p, prototype=None) and float(p.get("Conf", 0.0)) >= 0.20:
            enough_support = True
        if not enough_support:
            still_removed.append(p)
            continue

        cx = int(round(snap["GridX"]))
        cy = int(round(snap["GridY"]))
        ok, meta = roi_has_solder_ball(image_bgr, cx, cy, est_r, radius_ref=est_r)
        if not ok and not _candidate_detection_evidence(p, prototype=None):
            still_removed.append(p)
            continue

        new_p = dict(p)
        anchor_point, anchor_source = _resolve_node_anchor(
            node_key,
            grid_ctx,
            pitch_x,
            pitch_y,
            est_r=est_r,
            preferred_points=[p],
            raw_anchor_index=raw_anchor_index,
        )
        _apply_anchor_geometry(new_p, anchor_point, cx, cy, est_r, anchor_source=anchor_source)
        new_p["RecoveredByGrid"] = True
        new_p["AddedByGrid"] = False
        if ok:
            new_p.update(meta)
        else:
            new_p["RecoveredByNodeRule"] = True
        new_p.update(snap)
        new_p["OnValidGridNode"] = True
        new_p["InStandardGridComponent"] = node_key in set(grid_ctx.get("main_nodes", set()))
        current.append(new_p)
        occupancy[node_key] = new_p
        recovered.append(new_p)

    return current, still_removed, recovered


def grid_complete_matrix(points, soft_removed, image_bgr, pitch_x, pitch_y, est_r=None, tol_ratio: float = 0.28, grid_ctx: Optional[Dict[str, Any]] = None, raw_points: Optional[List[PointDict]] = None):
    if len(points) < 12:
        return points, soft_removed, [], [], {"matrix_completed": 0}

    if grid_ctx is None:
        _, grid_ctx, _ = build_standard_grid(points, pitch_x, pitch_y, tol_ratio=tol_ratio)
    if grid_ctx is None or not grid_ctx.get("valid_nodes"):
        return points, soft_removed, [], [], {"matrix_completed": 0}

    valid_nodes = set(grid_ctx["valid_nodes"])
    main_nodes = set(grid_ctx.get("main_nodes", set()))
    dense_grid_mode = _grid_ctx_is_dense(grid_ctx)
    all_points = assign_points_to_standard_grid(points, grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
    all_points = [dict(p) for p in all_points]
    occupancy = _collapse_points_to_grid_nodes(all_points)

    if est_r is None:
        est_r = float(np.median([p.get("Radius", 8.0) for p in all_points])) if all_points else 8.0
    raw_anchor_index = _build_raw_anchor_index(raw_points, grid_ctx, pitch_x, pitch_y, est_r=est_r)

    soft_by_node: Dict[Tuple[int, int], Tuple[int, PointDict]] = {}
    for i, p in enumerate(soft_removed):
        if _point_is_via_like(p, radius_ref=est_r):
            continue
        snap = snap_point_to_standard_grid(float(p["CenterX"]), float(p["CenterY"]), grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
        if not snap["OnStandardGrid"]:
            continue
        key = (int(snap["GridRow"]), int(snap["GridCol"]))
        if key not in valid_nodes:
            continue
        cand = dict(p)
        cand.update(snap)
        prev = soft_by_node.get(key)
        prev_res = float(prev[1].get("GridResidualNorm", 10.0)) if prev is not None else None
        cur_res = float(cand.get("GridResidualNorm", 10.0))
        if prev is None or cur_res < prev_res:
            soft_by_node[key] = (i, cand)

    recovered = []
    added = []
    used_soft = set()
    pending = set(valid_nodes) - set(occupancy.keys())
    changed = True
    scan_added = 0
    scan_recovered = 0

    while changed and pending:
        changed = False
        for node_key in sorted(list(pending)):
            axial, diag, has_lr, has_ud = _grid_node_neighbor_stats(set(occupancy.keys()), node_key)
            boundary_candidate = _grid_node_is_boundary(node_key, valid_nodes)
            if boundary_candidate:
                enough_support = axial >= 2 or has_lr or has_ud or (axial >= 1 and diag >= 2)
            else:
                enough_support = has_lr or has_ud or axial >= 2
            scan_support = False
            if dense_grid_mode:
                if boundary_candidate and (axial >= 1 and diag >= 1):
                    scan_support = True
                if (not boundary_candidate) and (axial >= 1 and diag >= 3):
                    scan_support = True
            if not enough_support:
                if not scan_support:
                    continue

            center = grid_node_to_image(node_key, grid_ctx)
            if center is None:
                continue
            cx = int(round(center[0]))
            cy = int(round(center[1]))
            local_est_r = _estimate_local_node_radius(occupancy, node_key, est_r)

            soft_item = soft_by_node.get(node_key)
            if soft_item is not None and soft_item[0] not in used_soft:
                ok, meta = roi_has_solder_ball(image_bgr, cx, cy, local_est_r, radius_ref=est_r)
                scan_ok = False
                scan_meta = {}
                if not ok:
                    scan_ok, scan_meta = scan_empty_grid_node(
                        image_bgr,
                        cx,
                        cy,
                        local_est_r,
                        radius_ref=est_r,
                        seed_conf=max(0.20, float(soft_item[1].get("Conf", 0.0))),
                    )
                cand = dict(soft_item[1])
                if ok or scan_ok or (boundary_candidate and not _point_is_via_like(cand, radius_ref=est_r)):
                    use_meta = scan_meta if scan_ok else meta
                    use_cx = int(round(use_meta.get("CenterX", cx)))
                    use_cy = int(round(use_meta.get("CenterY", cy)))
                    use_r = int(round(use_meta.get("Radius", local_est_r)))
                    if ok or scan_ok:
                        cand.update(use_meta)
                    anchor_point, anchor_source = _resolve_node_anchor(
                        node_key,
                        grid_ctx,
                        pitch_x,
                        pitch_y,
                        est_r=use_r,
                        preferred_points=[cand],
                        raw_anchor_index=raw_anchor_index,
                    )
                    _apply_anchor_geometry(cand, anchor_point, use_cx, use_cy, use_r, anchor_source=anchor_source)
                    cand["RecoveredByGrid"] = True
                    cand["RecoveredByMatrix"] = True
                    cand["AddedByGrid"] = False
                    cand["GridRow"] = int(node_key[0])
                    cand["GridCol"] = int(node_key[1])
                    cand["GridX"] = float(cand["CenterX"])
                    cand["GridY"] = float(cand["CenterY"])
                    cand["GridResidualPx"] = abs(float(cand["CenterX"]) - cx) + abs(float(cand["CenterY"]) - cy)
                    cand["GridResidualNorm"] = float(cand["GridResidualPx"]) / max(1e-6, min(pitch_x, pitch_y))
                    cand["OnStandardGrid"] = True
                    cand["OnValidGridNode"] = True
                    cand["InStandardGridComponent"] = node_key in main_nodes
                    all_points.append(cand)
                    occupancy[node_key] = cand
                    recovered.append(cand)
                    if scan_ok:
                        scan_recovered += 1
                    used_soft.add(soft_item[0])
                    pending.remove(node_key)
                    changed = True
                    continue

            ok, meta = roi_has_solder_ball(image_bgr, cx, cy, local_est_r, radius_ref=est_r)
            scan_ok = False
            scan_meta = {}
            if not ok:
                scan_ok, scan_meta = scan_empty_grid_node(
                    image_bgr,
                    cx,
                    cy,
                    local_est_r,
                    radius_ref=est_r,
                    seed_conf=0.22 if dense_grid_mode else 0.18,
                )
            if not ok and not scan_ok:
                continue
            use_meta = scan_meta if scan_ok else meta
            use_cx = int(round(use_meta.get("CenterX", cx)))
            use_cy = int(round(use_meta.get("CenterY", cy)))
            use_r = int(round(use_meta.get("Radius", local_est_r)))
            new_p = {
                "Conf": 0.0,
                "ClassId": -1,
                "AddedByGrid": True,
                "AddedByMatrix": True,
                "RecoveredByGrid": False,
                "CandidateSupport": int(axial + diag),
                "GridRow": int(node_key[0]),
                "GridCol": int(node_key[1]),
                "GridX": float(use_cx),
                "GridY": float(use_cy),
                "GridResidualPx": abs(use_cx - cx) + abs(use_cy - cy),
                "GridResidualNorm": float(abs(use_cx - cx) + abs(use_cy - cy)) / max(1e-6, min(pitch_x, pitch_y)),
                "OnStandardGrid": True,
                "OnValidGridNode": True,
                "InStandardGridComponent": node_key in main_nodes,
            }
            new_p.update(use_meta)
            anchor_point, anchor_source = _resolve_node_anchor(
                node_key,
                grid_ctx,
                pitch_x,
                pitch_y,
                est_r=use_r,
                preferred_points=None,
                raw_anchor_index=raw_anchor_index,
            )
            _apply_anchor_geometry(new_p, anchor_point, use_cx, use_cy, use_r, anchor_source=anchor_source)
            new_p["GridX"] = float(new_p["CenterX"])
            new_p["GridY"] = float(new_p["CenterY"])
            new_p["GridResidualPx"] = abs(float(new_p["CenterX"]) - cx) + abs(float(new_p["CenterY"]) - cy)
            new_p["GridResidualNorm"] = float(new_p["GridResidualPx"]) / max(1e-6, min(pitch_x, pitch_y))
            all_points.append(new_p)
            occupancy[node_key] = new_p
            added.append(new_p)
            if scan_ok:
                scan_added += 1
            pending.remove(node_key)
            changed = True

    kept_soft = [p for i, p in enumerate(soft_removed) if i not in used_soft]
    meta = {
        "matrix_completed": len(recovered) + len(added),
        "matrix_recovered": len(recovered),
        "matrix_added": len(added),
        "matrix_scan_recovered": int(scan_recovered),
        "matrix_scan_added": int(scan_added),
        "component_count": 1 if main_nodes else 0,
        "valid_node_count": int(len(valid_nodes)),
    }
    return all_points, kept_soft, recovered, added, meta


def grid_add_only_strict(points, image_bgr, pitch_x, pitch_y, tol_ratio=0.28, candidate_support=3, est_r=None, grid_ctx: Optional[Dict[str, Any]] = None, raw_points: Optional[List[PointDict]] = None):
    if len(points) < 12:
        return points, [], {"pitch_x": pitch_x, "pitch_y": pitch_y, "main_component_size": 0}

    if grid_ctx is None:
        _, grid_ctx, _ = build_standard_grid(points, pitch_x, pitch_y, tol_ratio=tol_ratio)
    if grid_ctx is None or not grid_ctx.get("valid_nodes"):
        return points, [], {"pitch_x": pitch_x, "pitch_y": pitch_y, "main_component_size": 0}

    valid_nodes = set(grid_ctx["valid_nodes"])
    main_nodes = set(grid_ctx.get("main_nodes", set()))
    all_points = assign_points_to_standard_grid(points, grid_ctx, on_grid_tol_ratio=max(0.32, tol_ratio + 0.08))
    all_points = [dict(p) for p in all_points]
    occupancy = _collapse_points_to_grid_nodes(all_points)

    if est_r is None:
        est_r = float(np.median([p.get("Radius", 8.0) for p in all_points])) if all_points else 8.0
    raw_anchor_index = _build_raw_anchor_index(raw_points, grid_ctx, pitch_x, pitch_y, est_r=est_r)

    added = []
    pending = set(valid_nodes) - set(occupancy.keys())
    changed = True

    while changed and pending:
        changed = False
        for node_key in sorted(list(pending)):
            boundary_candidate = _grid_node_is_boundary(node_key, valid_nodes)
            if boundary_candidate:
                continue
            axial, diag, has_lr, has_ud = _grid_node_neighbor_stats(set(occupancy.keys()), node_key)
            enough_support = (has_lr and has_ud) or axial >= max(3, candidate_support) or (axial >= 2 and diag >= 2)
            if not enough_support:
                continue

            center = grid_node_to_image(node_key, grid_ctx)
            if center is None:
                continue
            cx = int(round(center[0]))
            cy = int(round(center[1]))
            ok, meta = roi_has_solder_ball(image_bgr, cx, cy, est_r, radius_ref=est_r)
            if not ok:
                continue

            new_p = {
                "Conf": 0.0,
                "ClassId": -1,
                "AddedByGrid": True,
                "CandidateSupport": int(axial + diag),
                "GridRow": int(node_key[0]),
                "GridCol": int(node_key[1]),
                "GridX": float(cx),
                "GridY": float(cy),
                "GridResidualPx": 0.0,
                "GridResidualNorm": 0.0,
                "OnStandardGrid": True,
                "OnValidGridNode": True,
                "InStandardGridComponent": node_key in main_nodes,
            }
            new_p.update(meta)
            anchor_point, anchor_source = _resolve_node_anchor(
                node_key,
                grid_ctx,
                pitch_x,
                pitch_y,
                est_r=est_r,
                preferred_points=None,
                raw_anchor_index=raw_anchor_index,
            )
            _apply_anchor_geometry(new_p, anchor_point, cx, cy, est_r, anchor_source=anchor_source)
            new_p["GridX"] = float(new_p["CenterX"])
            new_p["GridY"] = float(new_p["CenterY"])
            new_p["GridResidualPx"] = abs(float(new_p["CenterX"]) - cx) + abs(float(new_p["CenterY"]) - cy)
            new_p["GridResidualNorm"] = float(new_p["GridResidualPx"]) / max(1e-6, min(pitch_x, pitch_y))
            all_points.append(new_p)
            occupancy[node_key] = new_p
            added.append(new_p)
            pending.remove(node_key)
            changed = True

    meta = {
        "pitch_x": float(pitch_x),
        "pitch_y": float(pitch_y),
        "main_component_size": len(main_nodes),
        "strict_valid_node_count": int(len(valid_nodes)),
    }
    return all_points, added, meta


def grid_add_only(points, image_bgr, pitch_x, pitch_y, tol_ratio=0.28, candidate_support=2, est_r=None):
    return grid_add_only_strict(points, image_bgr, pitch_x, pitch_y, tol_ratio=tol_ratio, candidate_support=max(candidate_support, 3), est_r=est_r)
