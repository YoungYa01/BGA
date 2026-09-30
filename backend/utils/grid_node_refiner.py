# -*- coding: utf-8 -*-
"""
BGA grid-node based second-stage refiner.

目标：不要再把问题建模成“检测所有圆然后过滤”，而是：
1) 从现有 YOLO/CV 候选点中估计主 BGA 阵列；
2) 生成规则 row/col 节点；
3) 对每个节点提取“焊点/过孔/背景”特征；
4) 保留焊点节点、删除强过孔/离网格点、补回漏检节点。

这个文件不依赖 sklearn/scipy，只需要 numpy + opencv，方便直接放进你现有项目的 utils/。

典型用法：
    from utils.grid_node_refiner import apply_grid_node_refine_to_result_item
    item = detector.infer_one("./test/1.jpg")
    item = apply_grid_node_refine_to_result_item(item)

输入点字段兼容你当前项目里的 PointDict：
    Left, Top, Right, Bottom, CenterX, CenterY, Radius, Width, Height, Conf, ClassId
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

PointDict = Dict[str, Any]


# -----------------------------
# 参数
# -----------------------------

@dataclass
class GridNodeRefineConfig:
    # pitch 估计
    min_points_for_grid: int = 16
    min_pitch_radius_ratio: float = 1.35
    max_pitch_radius_ratio: float = 8.0
    row_cross_tol_ratio: float = 0.65
    pitch_refine_tol_ratio: float = 0.28

    # 网格拟合 / 分配
    connect_tol_ratio: float = 0.36
    assign_tol_ratio: float = 0.43
    bbox_expand_pitch: float = 0.65
    max_grid_nodes: int = 6000

    # 多阵列支持：同一张图里可能有上下两个/多个 BGA 阵列。
    # v1 只保留最大连通阵列，容易把第二个真实阵列整片删掉。
    enable_multi_grid: bool = True
    max_grid_components: int = 8
    min_component_points: int = 12

    # 网格边界鲁棒收缩：少量边沿过孔/器件如果刚好落在 pitch 上，
    # 不应该把主 BGA 的 row/col 范围撑大。
    robust_grid_bounds: bool = True
    min_active_row_points: int = 3
    min_active_col_points: int = 3
    active_line_min_ratio: float = 0.22

    # 节点分类
    crop_radius_ratio: float = 1.75
    min_add_score: float = 3.10
    min_keep_score: float = 2.05
    strong_via_margin: float = 0.90
    strong_via_min_score: float = 3.00
    max_radius_ratio_for_via_delete: float = 0.92

    # 策略开关
    add_missing: bool = True
    remove_strong_via_on_grid: bool = True
    remove_offgrid: bool = True
    keep_uncertain_on_grid: bool = True
    keep_strong_solder_offgrid: bool = False

    # v2: 在密集阵列内部，宁愿少删一点，也不要把真实焊点删成红叉。
    protect_dense_on_grid: bool = True
    dense_neighbor_protect_min: int = 5
    dense_protect_max_residual_norm: float = 0.34
    dense_protect_radius_min: float = 0.68
    dense_protect_radius_max: float = 1.46

    # v2: 边沿过孔/大焊盘刚好贴到网格时，用尺寸 + 邻居支持做额外删除。
    remove_oversize_boundary_on_grid: bool = True
    boundary_neighbor_max: int = 4
    boundary_direct_neighbor_max: int = 2
    oversize_radius_ratio: float = 1.45

    # v2: BGA 阵列中间如果是空白/无焊点区域，不要因为“在网格上”就保留。
    remove_blank_on_grid: bool = True
    blank_max_fill_dark: float = 2.2
    blank_max_core_dark: float = 2.0
    blank_max_fill_dark_ratio: float = 0.30
    blank_max_solder_score: float = 2.15

    # v2: 补点必须有图像证据，并且只能补小缺口，不能把大面积 void 填满。
    require_add_visual_evidence: bool = True
    add_min_fill_dark: float = 3.0
    add_min_fill_dark_z: float = 0.16
    add_min_fill_dark_ratio: float = 0.36
    add_min_direct_neighbor: int = 2
    max_add_gap_nodes: int = 2

    # v3: 用“实心暗斑 blob”作为焊点视觉证据，而不是只相信 grid 邻居。
    # 这一步专门解决：大面积空白被补满、以及密集焊点/过孔混合图中漏补真实焊点。
    use_dark_blob_evidence: bool = True
    blob_min_area_ratio: float = 0.18        # dark blob area / (pi*r^2)
    blob_max_area_ratio: float = 1.45
    blob_min_circularity: float = 0.32
    blob_max_center_dist_norm: float = 0.48 # blob centroid distance / r
    blob_min_dark: float = 2.2              # outer_mean - blob_mean
    blob_min_fill_ratio: float = 0.24       # dark pixels inside fill disk

    # v3: 大 void 掩码。连续一大片“没有实心暗斑证据”的 grid nodes，视为非焊点区域。
    enable_void_mask: bool = True
    void_min_component_nodes: int = 9
    void_min_span_rows: int = 3
    void_min_span_cols: int = 3
    void_remove_existing: bool = True
    void_forbid_add: bool = True

    # v3: 强视觉证据可以跨越大缺口补点；弱证据仍然只能补小缺口。
    add_allow_large_gap_with_strong_visual: bool = True
    add_min_row_line_support: int = 3
    add_min_col_line_support: int = 3
    add_require_blob: bool = True

    # v3: 如果 grid 没拟合好，不要把非常像焊点的 off-grid 点一刀切删掉。
    keep_strong_visual_offgrid: bool = True

    # v4: 补点边界保护。边沿/外圈最容易把过孔、白色焊盘、空位置补成 BGA 焊点；
    # 默认不在 outer boundary 自动补点。真实边沿焊点通常会被一阶段检出，二阶段只负责保留。
    forbid_add_on_grid_boundary: bool = True
    forbid_add_boundary_margin: int = 1
    disable_large_gap_add_on_boundary: bool = True
    boundary_add_require_existing_candidate: bool = False
    boundary_add_min_solid_score: float = 3.8
    add_require_solid_evidence: bool = True

    # v4: void 判断改用“实心焊点证据”，而不是 v3 的 loose evidence。
    # 过孔、空白纹理、局部暗影即便有 blob，也不能打断 void。
    use_solid_evidence_for_void: bool = True
    solid_min_fill_dark: float = 2.8
    solid_min_core_dark: float = 1.4
    solid_min_fill_dark_ratio: float = 0.34
    solid_min_blob_fill_ratio: float = 0.22
    solid_min_blob_core_ratio: float = 0.08
    solid_max_core_bright_ratio: float = 0.22
    solid_max_ring_dark_ratio: float = 0.42
    solid_max_via_advantage: float = 0.55

    # v4: void 删除扩张。空白区内有零星误检时，它们会把 void 切碎；
    # 扩张一圈可以把边缘零星误检一起压住，但不动强实心焊点。
    void_dilate_iter: int = 1
    void_delete_loose_existing: bool = True

    # 调试
    attach_node_features: bool = True


@dataclass
class GridModel:
    grid_id: int
    pitch_x: float
    pitch_y: float
    phase_x: float
    phase_y: float
    min_col: int
    max_col: int
    min_row: int
    max_row: int
    radius_ref: float
    main_indices: List[int]
    bbox: Tuple[float, float, float, float]


# -----------------------------
# 基础工具
# -----------------------------

def _center(p: PointDict) -> Tuple[float, float]:
    return float(p["CenterX"]), float(p["CenterY"])


def _radius(p: PointDict) -> float:
    if "Radius" in p:
        return float(p["Radius"])
    return 0.5 * min(float(p["Right"] - p["Left"]), float(p["Bottom"] - p["Top"]))


def _robust_median(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    q1, q3 = np.percentile(arr, [20, 80])
    mid = arr[(arr >= q1) & (arr <= q3)]
    if mid.size == 0:
        mid = arr
    return float(np.median(mid))


def _clamp_box(left: int, top: int, right: int, bottom: int, w: int, h: int) -> Tuple[int, int, int, int]:
    left = max(0, min(int(left), w - 1))
    top = max(0, min(int(top), h - 1))
    right = max(left + 1, min(int(right), w))
    bottom = max(top + 1, min(int(bottom), h))
    return left, top, right, bottom


def _circular_phase(values: np.ndarray, period: float) -> float:
    """估计一组坐标在给定 pitch 下的公共相位。"""
    if period <= 1e-6 or values.size == 0:
        return 0.0
    angles = 2.0 * np.pi * (values % period) / period
    s = float(np.mean(np.sin(angles)))
    c = float(np.mean(np.cos(angles)))
    phase = (np.arctan2(s, c) * period / (2.0 * np.pi)) % period
    return float(phase)


def _normalize_index(values: np.ndarray, phase: float, pitch: float) -> np.ndarray:
    return np.rint((values - phase) / pitch).astype(np.int32)


# -----------------------------
# pitch / 主网格估计
# -----------------------------

def estimate_pitch_from_points(points: List[PointDict], cfg: GridNodeRefineConfig) -> Tuple[float, float, float]:
    """从候选点估计 pitch_x / pitch_y / radius_ref。

    思路：对每个点，在“近似同一行/列”的点里找最近邻距离，
    再对距离做鲁棒中位数。这个比直接 pairwise histogram 更不容易被 2*pitch、3*pitch 污染。
    """
    if not points:
        return 0.0, 0.0, 0.0

    xy = np.asarray([_center(p) for p in points], dtype=np.float32)
    radii = np.asarray([_radius(p) for p in points], dtype=np.float32)
    radii = radii[np.isfinite(radii) & (radii > 1)]
    radius_ref = _robust_median(radii, default=6.0)

    min_d = max(3.0, cfg.min_pitch_radius_ratio * radius_ref)
    max_d = max(min_d + 1.0, cfg.max_pitch_radius_ratio * radius_ref)
    cross_tol = max(3.0, cfg.row_cross_tol_ratio * radius_ref)

    xs, ys = xy[:, 0], xy[:, 1]
    dx_nearest: List[float] = []
    dy_nearest: List[float] = []

    for i in range(len(points)):
        dx = xs - xs[i]
        dy = ys - ys[i]

        # 同一行的右侧最近点
        mask_x = (dx > min_d) & (dx < max_d) & (np.abs(dy) <= cross_tol)
        if np.any(mask_x):
            dx_nearest.append(float(np.min(dx[mask_x])))

        # 同一列的下方最近点
        mask_y = (dy > min_d) & (dy < max_d) & (np.abs(dx) <= cross_tol)
        if np.any(mask_y):
            dy_nearest.append(float(np.min(dy[mask_y])))

    pitch_x = _refine_pitch(dx_nearest, cfg)
    pitch_y = _refine_pitch(dy_nearest, cfg)

    # 兜底：如果某一方向估计失败，用 2D 最近邻距离替代。
    if pitch_x <= 1e-6 or pitch_y <= 1e-6:
        nn = []
        for i in range(len(points)):
            dist = np.sqrt(np.sum((xy - xy[i]) ** 2, axis=1))
            dist = dist[(dist > min_d) & (dist < max_d)]
            if dist.size:
                nn.append(float(np.min(dist)))
        fallback = _refine_pitch(nn, cfg)
        if pitch_x <= 1e-6:
            pitch_x = fallback
        if pitch_y <= 1e-6:
            pitch_y = fallback

    return float(pitch_x), float(pitch_y), float(radius_ref)


def _refine_pitch(values: Iterable[float], cfg: GridNodeRefineConfig) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr) & (arr > 1)]
    if arr.size < 4:
        return 0.0
    med = float(np.median(arr))
    tol = max(2.0, cfg.pitch_refine_tol_ratio * med)
    near = arr[np.abs(arr - med) <= tol]
    if near.size >= 4:
        return float(np.median(near))
    return med


def _largest_component_by_grid_edges(
    points: List[PointDict], pitch_x: float, pitch_y: float, cfg: GridNodeRefineConfig
) -> List[int]:
    comps = _components_by_grid_edges(points, pitch_x, pitch_y, cfg)
    return comps[0] if comps else list(range(len(points)))


def _components_by_grid_edges(
    points: List[PointDict], pitch_x: float, pitch_y: float, cfg: GridNodeRefineConfig
) -> List[List[int]]:
    n = len(points)
    if n == 0:
        return []
    xy = np.asarray([_center(p) for p in points], dtype=np.float32)
    xs, ys = xy[:, 0], xy[:, 1]
    tol_x = max(3.0, cfg.connect_tol_ratio * pitch_x)
    tol_y = max(3.0, cfg.connect_tol_ratio * pitch_y)
    cross_x = max(3.0, 0.33 * pitch_x)
    cross_y = max(3.0, 0.33 * pitch_y)

    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        dx = xs[i + 1:] - xs[i]
        dy = ys[i + 1:] - ys[i]
        # 横向一跳邻居 / 纵向一跳邻居
        h = (np.abs(np.abs(dx) - pitch_x) <= tol_x) & (np.abs(dy) <= cross_y)
        v = (np.abs(np.abs(dy) - pitch_y) <= tol_y) & (np.abs(dx) <= cross_x)
        js = np.where(h | v)[0] + i + 1
        for j in js.tolist():
            adj[i].append(j)
            adj[j].append(i)

    visited = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(comp)

    comps.sort(key=len, reverse=True)
    return comps


def _largest_contiguous_span(values: List[int]) -> Tuple[int, int]:
    """从 active row/col 里找最长连续段。"""
    if not values:
        return 0, -1
    vals = sorted(set(int(v) for v in values))
    best_start = cur_start = vals[0]
    best_end = cur_end = vals[0]
    for v in vals[1:]:
        if v == cur_end + 1:
            cur_end = v
        else:
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = v
    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end
    return int(best_start), int(best_end)


def _robust_grid_span(indices: np.ndarray, min_points: int, ratio: float) -> Tuple[int, int]:
    """用 row/col 计数估计真实阵列边界，减少边沿过孔把范围撑大。"""
    if indices.size == 0:
        return 0, -1
    vals, counts = np.unique(indices.astype(np.int32), return_counts=True)
    if vals.size <= 2:
        return int(vals.min()), int(vals.max())
    med = float(np.median(counts))
    th = max(float(min_points), ratio * med)
    active = vals[counts >= th].astype(int).tolist()
    # 阈值过严时兜底，不要把小阵列吃掉。
    if len(active) < max(2, min(4, vals.size // 2)):
        active = vals[counts >= max(1.0, 0.5 * med)].astype(int).tolist()
    if not active:
        return int(vals.min()), int(vals.max())
    return _largest_contiguous_span(active)


def _make_grid_model_from_indices(
    points: List[PointDict],
    indices: List[int],
    pitch_x: float,
    pitch_y: float,
    radius_ref: float,
    cfg: GridNodeRefineConfig,
    grid_id: int = 0,
) -> Optional[GridModel]:
    if len(indices) < cfg.min_points_for_grid:
        return None

    main_xy = np.asarray([_center(points[i]) for i in indices], dtype=np.float32)
    xs, ys = main_xy[:, 0], main_xy[:, 1]

    phase_x = _circular_phase(xs, pitch_x)
    phase_y = _circular_phase(ys, pitch_y)

    cols = _normalize_index(xs, phase_x, pitch_x)
    rows = _normalize_index(ys, phase_y, pitch_y)

    if cfg.robust_grid_bounds:
        min_col, max_col = _robust_grid_span(cols, cfg.min_active_col_points, cfg.active_line_min_ratio)
        min_row, max_row = _robust_grid_span(rows, cfg.min_active_row_points, cfg.active_line_min_ratio)
    else:
        min_col, max_col = int(cols.min()), int(cols.max())
        min_row, max_row = int(rows.min()), int(rows.max())

    if min_col > max_col or min_row > max_row:
        return None

    # 边界用鲁棒 row/col 范围回推，避免被单个离群点撑大。
    x1 = float(phase_x + min_col * pitch_x - cfg.bbox_expand_pitch * pitch_x)
    x2 = float(phase_x + max_col * pitch_x + cfg.bbox_expand_pitch * pitch_x)
    y1 = float(phase_y + min_row * pitch_y - cfg.bbox_expand_pitch * pitch_y)
    y2 = float(phase_y + max_row * pitch_y + cfg.bbox_expand_pitch * pitch_y)

    node_count = (max_col - min_col + 1) * (max_row - min_row + 1)
    if node_count <= 0 or node_count > cfg.max_grid_nodes:
        return None

    return GridModel(
        grid_id=int(grid_id),
        pitch_x=float(pitch_x),
        pitch_y=float(pitch_y),
        phase_x=float(phase_x),
        phase_y=float(phase_y),
        min_col=int(min_col),
        max_col=int(max_col),
        min_row=int(min_row),
        max_row=int(max_row),
        radius_ref=float(radius_ref),
        main_indices=list(map(int, indices)),
        bbox=(x1, y1, x2, y2),
    )


def fit_grid_model(points: List[PointDict], cfg: Optional[GridNodeRefineConfig] = None) -> Optional[GridModel]:
    cfg = cfg or GridNodeRefineConfig()
    if len(points) < cfg.min_points_for_grid:
        return None

    pitch_x, pitch_y, radius_ref = estimate_pitch_from_points(points, cfg)
    if pitch_x <= 1e-6 or pitch_y <= 1e-6:
        return None

    main_indices = _largest_component_by_grid_edges(points, pitch_x, pitch_y, cfg)
    if len(main_indices) < cfg.min_points_for_grid:
        return None
    return _make_grid_model_from_indices(points, main_indices, pitch_x, pitch_y, radius_ref, cfg, grid_id=0)


def fit_grid_models(points: List[PointDict], cfg: Optional[GridNodeRefineConfig] = None) -> List[GridModel]:
    """拟合一个或多个 BGA 阵列。

    v1 只返回最大连通阵列；图里如果有上下两个真实焊点阵列，较小的那个会被
    当作 off_main_grid 删除。v2 会把每个规则连通分量单独拟合成 grid。
    """
    cfg = cfg or GridNodeRefineConfig()
    if len(points) < cfg.min_points_for_grid:
        return []

    if not cfg.enable_multi_grid:
        one = fit_grid_model(points, cfg)
        return [one] if one is not None else []

    pitch_x, pitch_y, radius_ref = estimate_pitch_from_points(points, cfg)
    if pitch_x <= 1e-6 or pitch_y <= 1e-6:
        return []

    comps = _components_by_grid_edges(points, pitch_x, pitch_y, cfg)
    models: List[GridModel] = []
    for comp in comps[: max(1, cfg.max_grid_components)]:
        if len(comp) < max(cfg.min_component_points, cfg.min_points_for_grid):
            continue
        m = _make_grid_model_from_indices(
            points, comp, pitch_x, pitch_y, radius_ref, cfg, grid_id=len(models)
        )
        if m is not None:
            models.append(m)

    # 如果组件划分过碎，退回单网格，避免直接失效。
    if not models:
        one = fit_grid_model(points, cfg)
        return [one] if one is not None else []
    return models


def iter_grid_nodes(grid: GridModel) -> Iterable[Tuple[int, int, float, float]]:
    for row in range(grid.min_row, grid.max_row + 1):
        y = grid.phase_y + row * grid.pitch_y
        for col in range(grid.min_col, grid.max_col + 1):
            x = grid.phase_x + col * grid.pitch_x
            yield row, col, float(x), float(y)


def assign_points_to_grid(
    points: List[PointDict], grid: GridModel, cfg: GridNodeRefineConfig
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, Tuple[int, int, float]]]:
    """把已有检测点分配到最近网格节点。

    返回：
        node_to_idx[(row, col)] = point_index
        idx_to_node[point_index] = (row, col, residual)
    """
    xy = np.asarray([_center(p) for p in points], dtype=np.float32)
    assign_tol = cfg.assign_tol_ratio * min(grid.pitch_x, grid.pitch_y)
    node_to_idx: Dict[Tuple[int, int], int] = {}
    idx_to_node: Dict[int, Tuple[int, int, float]] = {}

    # 先对每个点算最近 row/col，再处理冲突。
    candidates: List[Tuple[float, int, int, int]] = []
    for idx, (x, y) in enumerate(xy):
        col = int(round((float(x) - grid.phase_x) / grid.pitch_x))
        row = int(round((float(y) - grid.phase_y) / grid.pitch_y))
        if row < grid.min_row or row > grid.max_row or col < grid.min_col or col > grid.max_col:
            continue
        gx = grid.phase_x + col * grid.pitch_x
        gy = grid.phase_y + row * grid.pitch_y
        residual = float(np.hypot(float(x) - gx, float(y) - gy))
        if residual <= assign_tol:
            candidates.append((residual, idx, row, col))

    candidates.sort(key=lambda z: z[0])
    used_idx = set()
    used_node = set()
    for residual, idx, row, col in candidates:
        node = (row, col)
        if idx in used_idx or node in used_node:
            continue
        node_to_idx[node] = idx
        idx_to_node[idx] = (row, col, residual)
        used_idx.add(idx)
        used_node.add(node)

    return node_to_idx, idx_to_node


def assign_points_to_grid_models(
    points: List[PointDict], grids: List[GridModel], cfg: GridNodeRefineConfig
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[int, Tuple[int, int, int, float]]]:
    """把点分配到多个 grid 中最近的合法节点。

    返回：
        node_to_idx[(grid_id, row, col)] = point_index
        idx_to_node[point_index] = (grid_id, row, col, residual)
    """
    if not points or not grids:
        return {}, {}
    xy = np.asarray([_center(p) for p in points], dtype=np.float32)
    candidates: List[Tuple[float, int, int, int, int]] = []

    for idx, (x, y) in enumerate(xy):
        for grid in grids:
            assign_tol = cfg.assign_tol_ratio * min(grid.pitch_x, grid.pitch_y)
            col = int(round((float(x) - grid.phase_x) / grid.pitch_x))
            row = int(round((float(y) - grid.phase_y) / grid.pitch_y))
            if row < grid.min_row or row > grid.max_row or col < grid.min_col or col > grid.max_col:
                continue
            gx = grid.phase_x + col * grid.pitch_x
            gy = grid.phase_y + row * grid.pitch_y
            residual = float(np.hypot(float(x) - gx, float(y) - gy))
            if residual <= assign_tol:
                candidates.append((residual, idx, int(grid.grid_id), row, col))

    candidates.sort(key=lambda z: z[0])
    node_to_idx: Dict[Tuple[int, int, int], int] = {}
    idx_to_node: Dict[int, Tuple[int, int, int, float]] = {}
    used_idx = set()
    used_node = set()
    for residual, idx, gid, row, col in candidates:
        node = (gid, row, col)
        if idx in used_idx or node in used_node:
            continue
        node_to_idx[node] = idx
        idx_to_node[idx] = (gid, row, col, residual)
        used_idx.add(idx)
        used_node.add(node)
    return node_to_idx, idx_to_node


# -----------------------------
# 节点特征 / 分类
# -----------------------------

def _gray_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _radial_features(gray: np.ndarray, cx: float, cy: float, r: float) -> Dict[str, float]:
    h, w = gray.shape[:2]
    pad = int(max(6, round(1.75 * r)))
    l, t, rr, b = _clamp_box(int(cx - pad), int(cy - pad), int(cx + pad + 1), int(cy + pad + 1), w, h)
    roi = gray[t:b, l:rr]
    if roi.size == 0:
        return {"valid": 0.0}

    yy, xx = np.indices(roi.shape)
    xx = xx.astype(np.float32) + l
    yy = yy.astype(np.float32) + t
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    core = roi[dist <= 0.30 * r]
    fill = roi[dist <= 0.72 * r]
    ring = roi[(dist > 0.78 * r) & (dist <= 1.08 * r)]
    outer = roi[(dist > 1.16 * r) & (dist <= 1.62 * r)]

    if core.size < 6 or fill.size < 16 or ring.size < 8 or outer.size < 8:
        return {"valid": 0.0}

    core_m = float(np.mean(core))
    fill_m = float(np.mean(fill))
    ring_m = float(np.mean(ring))
    outer_m = float(np.mean(outer))
    outer_std = float(np.std(outer)) + 1e-6
    local_std = float(np.std(roi)) + 1e-6

    solder_fill_dark = outer_m - fill_m
    solder_core_dark = outer_m - core_m
    via_center_score = core_m - ring_m
    via_outer_score = outer_m - ring_m

    return {
        "valid": 1.0,
        "core_mean": core_m,
        "fill_mean": fill_m,
        "ring_mean": ring_m,
        "outer_mean": outer_m,
        "outer_std": outer_std,
        "local_std": local_std,
        "solder_fill_dark": solder_fill_dark,
        "solder_core_dark": solder_core_dark,
        "via_center_score": via_center_score,
        "via_outer_score": via_outer_score,
        "solder_fill_dark_z": solder_fill_dark / local_std,
        "solder_core_dark_z": solder_core_dark / local_std,
        "via_center_z": via_center_score / local_std,
        "via_outer_z": via_outer_score / local_std,
        "fill_dark_ratio": float(np.mean(fill <= (outer_m - max(2.0, 0.20 * local_std)))),
        "core_bright_ratio": float(np.mean(core >= (outer_m + max(2.0, 0.18 * local_std)))),
        "ring_dark_ratio": float(np.mean(ring <= (outer_m - max(2.0, 0.20 * local_std)))),
    }



def _dark_blob_features(gray: np.ndarray, cx: float, cy: float, r: float) -> Dict[str, float]:
    """检测节点中心附近是否存在“实心暗圆斑”。

    过孔常见是亮中心 + 暗环；焊点更像中心和填充区域一起变暗的实心 blob。
    因此补点/void 判断不能只看 grid 邻居，要看这里是否真的有一个居中的暗 blob。
    """
    h, w = gray.shape[:2]
    r = float(max(2.0, r))
    pad = int(max(8, round(1.85 * r)))
    l, t, rr, b = _clamp_box(int(cx - pad), int(cy - pad), int(cx + pad + 1), int(cy + pad + 1), w, h)
    roi = gray[t:b, l:rr]
    if roi.size == 0:
        return {"blob_valid": 0.0}

    yy, xx = np.indices(roi.shape)
    xx = xx.astype(np.float32) + l
    yy = yy.astype(np.float32) + t
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    outer = roi[(dist > 1.12 * r) & (dist <= 1.68 * r)]
    fill_mask_geom = dist <= 0.78 * r
    core_mask_geom = dist <= 0.36 * r
    if outer.size < 8 or np.count_nonzero(fill_mask_geom) < 12:
        return {"blob_valid": 0.0}

    outer_m = float(np.mean(outer))
    local_std = float(np.std(roi)) + 1e-6
    # 阈值不要太激进。真实焊点的暗斑通常会低于周围背景；空白区只有纹理时很难形成居中实心 blob。
    thr = outer_m - max(2.0, 0.16 * local_std)
    dark = (roi.astype(np.float32) <= thr).astype(np.uint8)
    # 限制在 1.15r 内，避免旁边走线/器件黑影连进来。
    dark[dist > 1.15 * r] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return {
            "blob_valid": 0.0,
            "blob_fill_ratio": float(np.mean(dark[fill_mask_geom] > 0)),
            "blob_core_ratio": float(np.mean(dark[core_mask_geom] > 0)) if np.count_nonzero(core_mask_geom) else 0.0,
        }

    roi_cx = float(cx - l)
    roi_cy = float(cy - t)
    best = None
    best_score = -1e9
    circle_area = float(np.pi * r * r)
    fill_area = float(np.count_nonzero(fill_mask_geom))
    core_area = float(np.count_nonzero(core_mask_geom))

    for lab in range(1, n):
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < max(3.0, 0.045 * circle_area) or area > 1.80 * circle_area:
            continue
        comp = labels == lab
        overlap_fill = float(np.count_nonzero(comp & fill_mask_geom)) / max(1.0, fill_area)
        overlap_core = float(np.count_nonzero(comp & core_mask_geom)) / max(1.0, core_area)
        ccx, ccy = float(cents[lab][0]), float(cents[lab][1])
        center_dist = float(np.hypot(ccx - roi_cx, ccy - roi_cy))
        if overlap_fill < 0.05 and center_dist > 0.70 * r:
            continue

        comp_u8 = comp.astype(np.uint8)
        cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        peri = float(cv2.arcLength(cnt, True))
        circularity = float(4.0 * np.pi * area / (peri * peri + 1e-6)) if peri > 1e-6 else 0.0
        blob_mean = float(np.mean(roi[comp]))
        blob_dark = outer_m - blob_mean
        area_ratio = area / max(1.0, circle_area)
        center_dist_norm = center_dist / max(1e-6, r)
        score = 2.4 * overlap_fill + 1.2 * overlap_core + 0.8 * circularity - 1.3 * center_dist_norm + 0.08 * blob_dark
        if score > best_score:
            best_score = score
            best = {
                "blob_valid": 1.0,
                "blob_area_ratio": float(area_ratio),
                "blob_circularity": float(circularity),
                "blob_center_dist_norm": float(center_dist_norm),
                "blob_dark": float(blob_dark),
                "blob_fill_ratio": float(overlap_fill),
                "blob_core_ratio": float(overlap_core),
            }

    if best is None:
        return {
            "blob_valid": 0.0,
            "blob_fill_ratio": float(np.mean(dark[fill_mask_geom] > 0)),
            "blob_core_ratio": float(np.mean(dark[core_mask_geom] > 0)) if np.count_nonzero(core_mask_geom) else 0.0,
        }
    return best


def _contour_features_for_node(gray: np.ndarray, cx: float, cy: float, r: float) -> Dict[str, float]:
    h, w = gray.shape[:2]
    pad = int(max(8, round(1.35 * r)))
    l, t, rr, b = _clamp_box(int(cx - pad), int(cy - pad), int(cx + pad + 1), int(cy + pad + 1), w, h)
    roi = gray[t:b, l:rr]
    if roi.size == 0:
        return {"contour_valid": 0.0}

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th_adapt = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ths = [
        cv2.morphologyEx(th_otsu, cv2.MORPH_OPEN, kernel, iterations=1),
        cv2.morphologyEx(th_adapt, cv2.MORPH_OPEN, kernel, iterations=1),
    ]

    best: Optional[Dict[str, float]] = None
    roi_cx = cx - l
    roi_cy = cy - t
    roi_area = float(roi.shape[0] * roi.shape[1])
    for th in ths:
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = float(cv2.contourArea(cnt))
            if area < roi_area * 0.025 or area > roi_area * 0.90:
                continue
            peri = float(cv2.arcLength(cnt, True))
            if peri <= 1e-6:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            m = cv2.moments(cnt)
            if abs(m["m00"]) < 1e-6:
                ccx, ccy = x + bw / 2.0, y + bh / 2.0
            else:
                ccx, ccy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            center_dist = float(np.hypot(ccx - roi_cx, ccy - roi_cy))
            pts = cnt[:, 0, :].astype(np.float32)
            rad = np.sqrt((pts[:, 0] - ccx) ** 2 + (pts[:, 1] - ccy) ** 2)
            feat = {
                "contour_valid": 1.0,
                "contour_area": area,
                "circularity": float(4.0 * np.pi * area / (peri * peri + 1e-6)),
                "extent": float(area / (bw * bh + 1e-6)),
                "shape_aspect": float(max(bw, bh) / (min(bw, bh) + 1e-6)),
                "radial_cv": float(np.std(rad) / (np.mean(rad) + 1e-6)),
                "contour_center_dist": center_dist,
                "contour_radius": float(0.5 * min(bw, bh)),
            }
            # 越圆、越居中越好。
            score = 2.0 * feat["circularity"] - feat["radial_cv"] - 0.06 * center_dist
            if best is None or score > best.get("_score", -1e9):
                feat["_score"] = float(score)
                best = feat

    if best is None:
        return {"contour_valid": 0.0}
    best.pop("_score", None)
    return best


def extract_node_features(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius_ref: float,
    grid: Optional[GridModel] = None,
    point: Optional[PointDict] = None,
    residual: Optional[float] = None,
    neighbor_support: int = 0,
) -> Dict[str, float]:
    gray = _gray_image(image_bgr)
    radius_ref = float(max(2.0, radius_ref))

    if point is not None:
        pr = _radius(point)
        radius_ratio = float(pr / radius_ref)
        use_r = float(np.clip(pr, 0.55 * radius_ref, 1.45 * radius_ref))
        conf = float(point.get("Conf", 0.0))
        box_aspect = float(max(point["Width"], point["Height"]) / (min(point["Width"], point["Height"]) + 1e-6)) \
            if "Width" in point and "Height" in point else 1.0
    else:
        radius_ratio = 1.0
        use_r = radius_ref
        conf = 0.0
        box_aspect = 1.0

    feat: Dict[str, float] = {
        "cx": float(cx),
        "cy": float(cy),
        "radius_ref": radius_ref,
        "radius_ratio": radius_ratio,
        "conf": conf,
        "box_aspect": box_aspect,
        "grid_residual": float(residual if residual is not None else 0.0),
        "grid_residual_norm": float((residual or 0.0) / max(1e-6, min(grid.pitch_x, grid.pitch_y))) if grid else 0.0,
        "neighbor_support": float(neighbor_support),
    }
    feat.update(_radial_features(gray, cx, cy, use_r))
    feat.update(_dark_blob_features(gray, cx, cy, use_r))
    feat.update(_contour_features_for_node(gray, cx, cy, use_r))
    return feat


def score_node(features: Dict[str, float]) -> Dict[str, Any]:
    """规则版节点分类器。

    后续如果你标注了 CSV，可以把这个函数替换成 LightGBM/RandomForest，
    但输入 features 的字段可以保持不变。
    """
    if features.get("valid", 0.0) < 0.5:
        return {
            "label": "unknown",
            "solder_score": 0.0,
            "via_score": 0.0,
            "reason": "invalid_radial_features",
        }

    fill_dark = float(features.get("solder_fill_dark", 0.0))
    core_dark = float(features.get("solder_core_dark", 0.0))
    fill_dark_z = float(features.get("solder_fill_dark_z", 0.0))
    core_dark_z = float(features.get("solder_core_dark_z", 0.0))
    fill_dark_ratio = float(features.get("fill_dark_ratio", 0.0))

    via_center = float(features.get("via_center_score", 0.0))
    via_outer = float(features.get("via_outer_score", 0.0))
    via_center_z = float(features.get("via_center_z", 0.0))
    via_outer_z = float(features.get("via_outer_z", 0.0))
    core_bright_ratio = float(features.get("core_bright_ratio", 0.0))
    ring_dark_ratio = float(features.get("ring_dark_ratio", 0.0))

    radius_ratio = float(features.get("radius_ratio", 1.0))
    circularity = float(features.get("circularity", 0.0))
    radial_cv = float(features.get("radial_cv", 0.8))
    box_aspect = float(features.get("box_aspect", 1.0))
    neighbor_support = float(features.get("neighbor_support", 0.0))
    residual_norm = float(features.get("grid_residual_norm", 0.0))
    conf = float(features.get("conf", 0.0))

    solder_score = 0.0
    via_score = 0.0
    reasons: List[str] = []

    # 焊点：内部整体更暗、中心也暗、形态圆、尺寸接近主焊点、网格支持强。
    if fill_dark >= 2.5 or fill_dark_z >= 0.18:
        solder_score += 1.00
        reasons.append("fill_dark")
    if core_dark >= 1.8 or core_dark_z >= 0.13:
        solder_score += 0.85
        reasons.append("core_dark")
    if fill_dark_ratio >= 0.38:
        solder_score += 0.75
        reasons.append("fill_dark_ratio")
    if 0.76 <= radius_ratio <= 1.35:
        solder_score += 0.65
        reasons.append("radius_ok")
    if circularity >= 0.52 and radial_cv <= 0.36:
        solder_score += 0.55
        reasons.append("shape_round")
    if box_aspect <= 1.35:
        solder_score += 0.30
    if conf >= 0.25:
        solder_score += 0.25
    # 网格支持是这里的重点：一个弱暗核在标准网格节点上，也比离网格圆更像焊点。
    solder_score += min(1.00, 0.24 * neighbor_support)
    if residual_norm <= 0.22:
        solder_score += 0.50
    elif residual_norm <= 0.36:
        solder_score += 0.22

    # 过孔：亮中心 + 暗环；通常更小，或填充暗度弱。
    if via_center >= 6.0 or via_center_z >= 0.35:
        via_score += 1.05
        reasons.append("via_center_bright")
    if via_outer >= 3.5 or via_outer_z >= 0.25:
        via_score += 0.90
        reasons.append("via_dark_ring")
    if core_bright_ratio >= 0.25:
        via_score += 0.65
        reasons.append("core_bright_ratio")
    if ring_dark_ratio >= 0.32:
        via_score += 0.55
        reasons.append("ring_dark_ratio")
    if radius_ratio <= 0.88:
        via_score += 0.70
        reasons.append("small_radius")
    if fill_dark < 2.0 and fill_dark_ratio < 0.32:
        via_score += 0.40
        reasons.append("weak_solder_fill")

    # 明显长条/矩形，两个分数都不应该高；偏向 non_solder。
    if box_aspect >= 1.65:
        solder_score -= 1.00
        via_score += 0.40
        reasons.append("box_aspect_bad")
    if circularity > 0 and circularity < 0.34:
        solder_score -= 0.70
        reasons.append("low_circularity")

    if via_score >= solder_score + 0.55 and via_score >= 2.4:
        label = "via"
    elif solder_score >= 2.05 and solder_score >= via_score - 0.25:
        label = "solder"
    else:
        label = "unknown"

    return {
        "label": label,
        "solder_score": float(solder_score),
        "via_score": float(via_score),
        "reason": "+".join(reasons[:8]),
    }


def _neighbor_support(node_to_idx: Dict[Tuple[int, int], int], row: int, col: int) -> int:
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        if (row + dr, col + dc) in node_to_idx:
            count += 1
    return count


def _neighbor_support_multi(node_to_idx: Dict[Tuple[int, int, int], int], gid: int, row: int, col: int) -> int:
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        if (gid, row + dr, col + dc) in node_to_idx:
            count += 1
    return count


def _direct_neighbor_support_multi(node_to_idx: Dict[Tuple[int, int, int], int], gid: int, row: int, col: int) -> int:
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if (gid, row + dr, col + dc) in node_to_idx:
            count += 1
    return count


def _is_boundary_node(grid: GridModel, row: int, col: int) -> bool:
    return row in (grid.min_row, grid.max_row) or col in (grid.min_col, grid.max_col)


def _missing_gap_len(node_to_idx: Dict[Tuple[int, int, int], int], gid: int, row: int, col: int, axis: str) -> int:
    """计算某个缺失 node 所在的连续缺失长度，用来避免填大面积 void。"""
    length = 1
    if axis == "h":
        for step in (-1, 1):
            c = col + step
            while (gid, row, c) not in node_to_idx:
                length += 1
                c += step
                if length > 50:
                    break
    else:
        for step in (-1, 1):
            r = row + step
            while (gid, r, col) not in node_to_idx:
                length += 1
                r += step
                if length > 50:
                    break
    return int(length)


def _missing_gap_len_bounded(
    node_to_idx: Dict[Tuple[int, int, int], int], grid: GridModel, row: int, col: int, axis: str
) -> int:
    """在 grid 边界内计算连续缺失长度。"""
    gid = int(grid.grid_id)
    length = 1
    if axis == "h":
        c = col - 1
        while c >= grid.min_col and (gid, row, c) not in node_to_idx:
            length += 1
            c -= 1
        c = col + 1
        while c <= grid.max_col and (gid, row, c) not in node_to_idx:
            length += 1
            c += 1
    else:
        r = row - 1
        while r >= grid.min_row and (gid, r, col) not in node_to_idx:
            length += 1
            r -= 1
        r = row + 1
        while r <= grid.max_row and (gid, r, col) not in node_to_idx:
            length += 1
            r += 1
    return int(length)


def _dense_grid_protected(features: Dict[str, float], score: Dict[str, Any], cfg: GridNodeRefineConfig) -> bool:
    """密集 BGA 内部的点，即便有一点 via-like，也优先保护，解决图一误删。"""
    if not cfg.protect_dense_on_grid:
        return False
    ns = float(features.get("neighbor_support", 0.0))
    residual_norm = float(features.get("grid_residual_norm", 1.0))
    radius_ratio = float(features.get("radius_ratio", 1.0))
    box_aspect = float(features.get("box_aspect", 1.0))
    solder_score = float(score.get("solder_score", 0.0))
    return (
        ns >= cfg.dense_neighbor_protect_min
        and residual_norm <= cfg.dense_protect_max_residual_norm
        and cfg.dense_protect_radius_min <= radius_ratio <= cfg.dense_protect_radius_max
        and box_aspect <= 1.45
        and solder_score >= 1.25
    )


def _blank_like_on_grid(features: Dict[str, float], score: Dict[str, Any], cfg: GridNodeRefineConfig) -> bool:
    """在网格上但图像证据很弱，倾向于空白/背景。"""
    if not cfg.remove_blank_on_grid:
        return False
    fill_dark = float(features.get("solder_fill_dark", 0.0))
    core_dark = float(features.get("solder_core_dark", 0.0))
    fill_dark_ratio = float(features.get("fill_dark_ratio", 0.0))
    solder_score = float(score.get("solder_score", 0.0))
    via_score = float(score.get("via_score", 0.0))
    return (
        fill_dark <= cfg.blank_max_fill_dark
        and core_dark <= cfg.blank_max_core_dark
        and fill_dark_ratio <= cfg.blank_max_fill_dark_ratio
        and solder_score <= cfg.blank_max_solder_score
        and via_score < 2.3
    )



def _dark_blob_ok(features: Dict[str, float], cfg: GridNodeRefineConfig) -> bool:
    if not cfg.use_dark_blob_evidence:
        return True
    if float(features.get("blob_valid", 0.0)) < 0.5:
        return False
    area_ratio = float(features.get("blob_area_ratio", 0.0))
    circularity = float(features.get("blob_circularity", 0.0))
    center_dist_norm = float(features.get("blob_center_dist_norm", 9.0))
    blob_dark = float(features.get("blob_dark", 0.0))
    fill_ratio = float(features.get("blob_fill_ratio", 0.0))
    return (
        cfg.blob_min_area_ratio <= area_ratio <= cfg.blob_max_area_ratio
        and circularity >= cfg.blob_min_circularity
        and center_dist_norm <= cfg.blob_max_center_dist_norm
        and blob_dark >= cfg.blob_min_dark
        and fill_ratio >= cfg.blob_min_fill_ratio
    )


def _visual_solder_evidence(features: Dict[str, float], score: Dict[str, Any], cfg: GridNodeRefineConfig, strong: bool = False) -> bool:
    """不依赖 grid 邻居的焊点视觉证据。

    strong=False 用于 void 判断，稍宽松；strong=True 用于补点，必须更像实心暗焊点。
    """
    if features.get("valid", 0.0) < 0.5:
        return False
    fill_dark = float(features.get("solder_fill_dark", 0.0))
    core_dark = float(features.get("solder_core_dark", 0.0))
    fill_dark_z = float(features.get("solder_fill_dark_z", 0.0))
    fill_dark_ratio = float(features.get("fill_dark_ratio", 0.0))
    core_bright_ratio = float(features.get("core_bright_ratio", 0.0))
    ring_dark_ratio = float(features.get("ring_dark_ratio", 0.0))
    via_score = float(score.get("via_score", 0.0))
    solder_score = float(score.get("solder_score", 0.0))
    blob_ok = _dark_blob_ok(features, cfg)

    # 过孔典型：中心亮、环暗。即便落在 grid 上，也不能作为补点证据。
    strong_via_shape = (core_bright_ratio >= 0.24 and ring_dark_ratio >= 0.28 and via_score >= solder_score + 0.45)
    if strong_via_shape:
        return False

    if strong:
        radial_ok = (
            (fill_dark >= cfg.add_min_fill_dark or fill_dark_z >= cfg.add_min_fill_dark_z)
            and fill_dark_ratio >= cfg.add_min_fill_dark_ratio
            and core_dark >= max(1.2, 0.45 * cfg.add_min_fill_dark)
        )
        if cfg.add_require_blob:
            return bool(radial_ok and blob_ok and via_score <= solder_score + 0.25)
        return bool(radial_ok and via_score <= solder_score + 0.25)

    # void 判断用宽松证据：只要这里像一个真实暗焊点，就不要把它归入 void。
    radial_loose = (
        (fill_dark >= max(1.8, 0.62 * cfg.add_min_fill_dark) or fill_dark_z >= max(0.10, 0.62 * cfg.add_min_fill_dark_z))
        and fill_dark_ratio >= max(0.24, 0.70 * cfg.add_min_fill_dark_ratio)
    )
    return bool((radial_loose or blob_ok) and via_score <= solder_score + 0.85)




def _solid_solder_evidence(features: Dict[str, float], score: Dict[str, Any], cfg: GridNodeRefineConfig) -> bool:
    """更严格的“实心焊点”证据。

    v3 的 visual_loose 适合保护真实焊点，但用于 void 会太宽：
    过孔的暗环、空白区纹理、器件阴影都可能让 visual_loose=True。
    v4 用 solid evidence 来决定一个节点是否足以打断/阻止 void。
    """
    if features.get("valid", 0.0) < 0.5:
        return False

    fill_dark = float(features.get("solder_fill_dark", 0.0))
    core_dark = float(features.get("solder_core_dark", 0.0))
    fill_dark_z = float(features.get("solder_fill_dark_z", 0.0))
    fill_dark_ratio = float(features.get("fill_dark_ratio", 0.0))
    core_bright_ratio = float(features.get("core_bright_ratio", 0.0))
    ring_dark_ratio = float(features.get("ring_dark_ratio", 0.0))
    via_score = float(score.get("via_score", 0.0))
    solder_score = float(score.get("solder_score", 0.0))
    blob_valid = float(features.get("blob_valid", 0.0)) >= 0.5
    blob_fill_ratio = float(features.get("blob_fill_ratio", 0.0))
    blob_core_ratio = float(features.get("blob_core_ratio", 0.0))
    blob_center_dist_norm = float(features.get("blob_center_dist_norm", 9.0))
    blob_dark = float(features.get("blob_dark", 0.0))

    # 过孔/环状结构：中心亮或暗环太强，不能算 solid solder。
    if core_bright_ratio >= cfg.solid_max_core_bright_ratio and ring_dark_ratio >= 0.26:
        return False
    if via_score >= solder_score + cfg.solid_max_via_advantage:
        return False
    if ring_dark_ratio >= cfg.solid_max_ring_dark_ratio and core_dark < cfg.solid_min_core_dark:
        return False

    radial_solid = (
        (fill_dark >= cfg.solid_min_fill_dark or fill_dark_z >= max(0.12, cfg.add_min_fill_dark_z))
        and core_dark >= cfg.solid_min_core_dark
        and fill_dark_ratio >= cfg.solid_min_fill_dark_ratio
    )
    blob_solid = (
        blob_valid
        and blob_dark >= max(1.6, 0.75 * cfg.blob_min_dark)
        and blob_fill_ratio >= cfg.solid_min_blob_fill_ratio
        and blob_core_ratio >= cfg.solid_min_blob_core_ratio
        and blob_center_dist_norm <= max(0.55, cfg.blob_max_center_dist_norm)
    )
    return bool((radial_solid or (radial_solid and blob_solid) or (blob_solid and fill_dark_ratio >= 0.28))
                and via_score <= solder_score + cfg.solid_max_via_advantage)


def _is_boundary_margin_node(grid: GridModel, row: int, col: int, margin: int) -> bool:
    margin = max(0, int(margin))
    return (
        row <= grid.min_row + margin
        or row >= grid.max_row - margin
        or col <= grid.min_col + margin
        or col >= grid.max_col - margin
    )


def _boundary_add_allowed(
    grid: GridModel,
    row: int,
    col: int,
    features: Dict[str, float],
    score: Dict[str, Any],
    visual_strong: bool,
    direct_ns: int,
    cfg: GridNodeRefineConfig,
) -> bool:
    """边界缺失节点是否允许补点。

    边界是最容易误补的位置：grid 会被外围过孔/焊盘撑到外圈，
    然后把外圈空位置和过孔当作缺失焊点补进去。默认直接禁止；
    如果关闭禁止，也要求非常强的 solid evidence。 
    """
    if not _is_boundary_margin_node(grid, row, col, cfg.forbid_add_boundary_margin):
        return True
    if cfg.forbid_add_on_grid_boundary:
        return False
    if not visual_strong:
        return False
    if direct_ns < max(2, cfg.add_min_direct_neighbor):
        return False
    if not _solid_solder_evidence(features, score, cfg):
        return False
    return float(score.get("solder_score", 0.0)) >= cfg.boundary_add_min_solid_score

def _compute_grid_node_cache(
    image_bgr: np.ndarray,
    points: List[PointDict],
    grids: List[GridModel],
    node_to_idx: Dict[Tuple[int, int, int], int],
    idx_to_node: Dict[int, Tuple[int, int, int, float]],
    cfg: GridNodeRefineConfig,
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """预先计算每个 grid node 的视觉证据，用于 void mask 和补点。"""
    cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for grid in grids:
        gid = int(grid.grid_id)
        for row, col, gx, gy in iter_grid_nodes(grid):
            node = (gid, row, col)
            idx = node_to_idx.get(node)
            if idx is not None:
                p = points[idx]
                cx, cy = _center(p)
                residual = idx_to_node[idx][3]
                point = p
            else:
                cx, cy = gx, gy
                residual = 0.0
                point = None
            ns = _neighbor_support_multi(node_to_idx, gid, row, col)
            direct_ns = _direct_neighbor_support_multi(node_to_idx, gid, row, col)
            features = extract_node_features(
                image_bgr, cx, cy, grid.radius_ref, grid=grid, point=point, residual=residual, neighbor_support=ns
            )
            score = score_node(features)
            visual_loose = _visual_solder_evidence(features, score, cfg, strong=False)
            visual_strong = _visual_solder_evidence(features, score, cfg, strong=True)
            cache[node] = {
                "grid": grid,
                "row": int(row),
                "col": int(col),
                "x": float(gx),
                "y": float(gy),
                "idx": idx,
                "features": features,
                "score": score,
                "neighbor_support": int(ns),
                "direct_neighbor_support": int(direct_ns),
                "visual_loose": bool(visual_loose),
                "visual_strong": bool(visual_strong),
                "visual_solid": bool(_solid_solder_evidence(features, score, cfg)),
            }
    return cache


def _compute_void_nodes(
    grids: List[GridModel],
    node_cache: Dict[Tuple[int, int, int], Dict[str, Any]],
    cfg: GridNodeRefineConfig,
) -> set:
    """找连续大块非焊点区域，阻止 grid 把它们补成焊点。

    v4 关键修正：用 solid_solder_evidence 判断“这个节点是否真像实心焊点”。
    v3 用 visual_loose，过孔暗环/空白纹理可能也会 loose=True，导致 void 被切碎。
    """
    void_nodes = set()
    if not cfg.enable_void_mask:
        return void_nodes

    solid_nodes = set()
    for node, rec in node_cache.items():
        if cfg.use_solid_evidence_for_void:
            if _solid_solder_evidence(rec.get("features", {}), rec.get("score", {}), cfg):
                solid_nodes.add(node)
        else:
            if rec.get("visual_loose", False):
                solid_nodes.add(node)

    for grid in grids:
        gid = int(grid.grid_id)
        visited = set()
        for row in range(grid.min_row, grid.max_row + 1):
            for col in range(grid.min_col, grid.max_col + 1):
                node = (gid, row, col)
                if node in visited:
                    continue
                rec = node_cache.get(node)
                if rec is None:
                    continue
                if node in solid_nodes:
                    visited.add(node)
                    continue

                stack = [node]
                visited.add(node)
                comp = []
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    _, r, c = u
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        v = (gid, nr, nc)
                        if v in visited:
                            continue
                        if nr < grid.min_row or nr > grid.max_row or nc < grid.min_col or nc > grid.max_col:
                            continue
                        rv = node_cache.get(v)
                        if rv is None:
                            continue
                        if v in solid_nodes:
                            visited.add(v)
                            continue
                        visited.add(v)
                        stack.append(v)

                if len(comp) < cfg.void_min_component_nodes:
                    continue
                rows = [n[1] for n in comp]
                cols = [n[2] for n in comp]
                span_r = max(rows) - min(rows) + 1
                span_c = max(cols) - min(cols) + 1
                if span_r >= cfg.void_min_span_rows and span_c >= cfg.void_min_span_cols:
                    void_nodes.update(comp)

    # v4: 对 void 做轻微扩张，吞掉 void 边上的零星误检/过孔。
    # 但 strong solid 节点不扩进去，避免吃掉真实焊点边界。
    for _ in range(max(0, int(cfg.void_dilate_iter))):
        extra = set()
        for gid, row, col in list(void_nodes):
            grid = next((g for g in grids if int(g.grid_id) == int(gid)), None)
            if grid is None:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                v = (gid, nr, nc)
                if nr < grid.min_row or nr > grid.max_row or nc < grid.min_col or nc > grid.max_col:
                    continue
                rec = node_cache.get(v)
                if rec is None:
                    continue
                if v in solid_nodes:
                    continue
                extra.add(v)
        void_nodes.update(extra)

    return void_nodes

def _line_visual_support(
    node_cache: Dict[Tuple[int, int, int], Dict[str, Any]], grid: GridModel, row: int, col: int
) -> Tuple[int, int]:
    gid = int(grid.grid_id)
    row_support = 0
    for c in range(grid.min_col, grid.max_col + 1):
        if c == col:
            continue
        rec = node_cache.get((gid, row, c))
        if rec and rec.get("visual_solid", False):
            row_support += 1
    col_support = 0
    for r in range(grid.min_row, grid.max_row + 1):
        if r == row:
            continue
        rec = node_cache.get((gid, r, col))
        if rec and rec.get("visual_solid", False):
            col_support += 1
    return int(row_support), int(col_support)


def _oversize_boundary_like(
    grid: GridModel,
    row: int,
    col: int,
    features: Dict[str, float],
    direct_ns: int,
    all_ns: int,
    cfg: GridNodeRefineConfig,
) -> bool:
    """边沿大圆/白色过孔焊盘刚好落在网格上时的额外删除规则。"""
    if not cfg.remove_oversize_boundary_on_grid:
        return False
    radius_ratio = float(features.get("radius_ratio", 1.0))
    if radius_ratio < cfg.oversize_radius_ratio:
        return False
    return _is_boundary_node(grid, row, col) or all_ns <= cfg.boundary_neighbor_max or direct_ns <= cfg.boundary_direct_neighbor_max


def _has_add_visual_evidence(features: Dict[str, float], score: Dict[str, Any], cfg: GridNodeRefineConfig) -> bool:
    """缺失节点补点时的硬证据门槛。

    v3 不再只看 radial 均值；必须有实心暗 blob 或者非常强的实心焊点证据。
    这样大面积空白/过孔环不会被 grid 自动填满。
    """
    if not cfg.require_add_visual_evidence:
        return True
    if not _visual_solder_evidence(features, score, cfg, strong=True):
        return False
    solder_score = float(score.get("solder_score", 0.0))
    return solder_score >= max(1.75, cfg.min_add_score - 0.70)


def _make_added_point(cx: float, cy: float, r: float, score: Dict[str, Any], features: Dict[str, float]) -> PointDict:
    left = int(round(cx - r))
    top = int(round(cy - r))
    right = int(round(cx + r))
    bottom = int(round(cy + r))
    p: PointDict = {
        "Left": left,
        "Top": top,
        "Right": right,
        "Bottom": bottom,
        "CenterX": float(cx),
        "CenterY": float(cy),
        "Radius": float(r),
        "Width": int(max(1, right - left)),
        "Height": int(max(1, bottom - top)),
        "Conf": float(min(0.99, max(0.01, score.get("solder_score", 0.0) / 5.5))),
        "ClassId": 0,
        "AddedByGrid": True,
        "AddedByGridNode": True,
        "GridNodeLabel": score.get("label", "solder"),
        "GridNodeSolderScore": round(float(score.get("solder_score", 0.0)), 4),
        "GridNodeViaScore": round(float(score.get("via_score", 0.0)), 4),
        "GridNodeReason": score.get("reason", ""),
    }
    for k in ["solder_fill_dark", "solder_core_dark", "fill_dark_ratio", "via_center_score", "via_outer_score"]:
        if k in features:
            p[k] = round(float(features[k]), 4)
    return p


# -----------------------------
# 主入口：refine
# -----------------------------

def refine_points_by_grid_nodes(
    image_bgr: np.ndarray,
    points: List[PointDict],
    cfg: Optional[GridNodeRefineConfig] = None,
) -> Tuple[List[PointDict], List[PointDict], List[PointDict], Dict[str, Any]]:
    """输入当前 points，输出：kept_points, removed_points, added_points, meta。

    v3 关键变化：
      1) 每个 grid node 都先做“实心暗斑”视觉证据判断；
      2) 连续大块没有实心焊点证据的节点会形成 void mask，禁止补点，并可删除弱证据误检；
      3) 对密集焊点/过孔混合图，强视觉证据可以跨越大缺口补点，不再只补 1~2 个小缺口；
      4) grid 拟合不稳时，off-grid 但强视觉像焊点的点默认保留，避免整片漏删。
    """
    cfg = cfg or GridNodeRefineConfig()
    if image_bgr is None or len(points) < cfg.min_points_for_grid:
        return points, [], [], {"enabled": False, "reason": "not_enough_points"}

    grids = fit_grid_models(points, cfg)
    if not grids:
        return points, [], [], {"enabled": False, "reason": "grid_fit_failed"}

    grid_by_id = {int(g.grid_id): g for g in grids}
    node_to_idx, idx_to_node = assign_points_to_grid_models(points, grids, cfg)

    # v3: 先给所有 grid node 建缓存，再从缓存里识别 void。后续删除/补点都参考它。
    node_cache = _compute_grid_node_cache(image_bgr, points, grids, node_to_idx, idx_to_node, cfg)
    void_nodes = _compute_void_nodes(grids, node_cache, cfg)

    kept: List[PointDict] = []
    removed: List[PointDict] = []
    added: List[PointDict] = []

    # 先处理已有点：在任意 grid 上的点做节点分类；离所有 grid 的点默认更可疑，
    # 但 v3 会保留强视觉焊点，避免图二这种复杂密集图被误删太多。
    for idx, p in enumerate(points):
        cx, cy = _center(p)
        assigned = idx in idx_to_node

        if assigned:
            gid, row, col, residual = idx_to_node[idx]
            grid = grid_by_id[gid]
            node = (gid, row, col)
            rec = node_cache.get(node)
            if rec is not None:
                features = rec["features"]
                score = rec["score"]
                ns = int(rec["neighbor_support"])
                direct_ns = int(rec["direct_neighbor_support"])
                visual_loose = bool(rec["visual_loose"])
                visual_strong = bool(rec["visual_strong"])
                visual_solid = bool(rec.get("visual_solid", False))
            else:
                ns = _neighbor_support_multi(node_to_idx, gid, row, col)
                direct_ns = _direct_neighbor_support_multi(node_to_idx, gid, row, col)
                features = extract_node_features(
                    image_bgr, cx, cy, grid.radius_ref, grid=grid, point=p, residual=residual, neighbor_support=ns
                )
                score = score_node(features)
                visual_loose = _visual_solder_evidence(features, score, cfg, strong=False)
                visual_strong = _visual_solder_evidence(features, score, cfg, strong=True)
                visual_solid = _solid_solder_evidence(features, score, cfg)

            q = dict(p)
            q["GridId"] = int(gid)
            q["GridRow"] = int(row)
            q["GridCol"] = int(col)
            q["GridResidual"] = round(float(residual), 4)
            q["GridDirectNeighborSupport"] = int(direct_ns)
            q["GridNodeLabel"] = score["label"]
            q["GridNodeSolderScore"] = round(float(score["solder_score"]), 4)
            q["GridNodeViaScore"] = round(float(score["via_score"]), 4)
            q["GridNodeReason"] = score["reason"]
            q["GridNodeVisualLoose"] = bool(visual_loose)
            q["GridNodeVisualStrong"] = bool(visual_strong)
            q["GridNodeVisualSolid"] = bool(visual_solid)
            q["GridNodeInVoid"] = bool(node in void_nodes)
            if cfg.attach_node_features:
                q["GridNodeFeatures"] = _round_feature_dict(features)

            dense_protected = _dense_grid_protected(features, score, cfg)
            blank_like = _blank_like_on_grid(features, score, cfg)
            oversize_boundary = _oversize_boundary_like(grid, row, col, features, direct_ns, ns, cfg)

            strong_via = (
                score["label"] == "via"
                and score["via_score"] >= cfg.strong_via_min_score
                and score["via_score"] >= score["solder_score"] + cfg.strong_via_margin
            )

            # v3: 大 void 内的弱证据点优先删除，dense_protected 不能覆盖 void。
            if cfg.enable_void_mask and cfg.void_remove_existing and node in void_nodes and not visual_solid:
                q["RejectReason"] = "grid_node_void_non_solid_evidence"
                removed.append(q)
            elif cfg.remove_oversize_boundary_on_grid and oversize_boundary and not dense_protected and not visual_strong:
                q["RejectReason"] = "grid_node_oversize_boundary"
                removed.append(q)
            elif cfg.remove_strong_via_on_grid and strong_via and not dense_protected and not visual_strong:
                q["RejectReason"] = "grid_node_strong_via"
                removed.append(q)
            elif cfg.remove_blank_on_grid and blank_like and not dense_protected and not visual_loose:
                q["RejectReason"] = "grid_node_blank_like"
                removed.append(q)
            elif score["label"] == "solder" or cfg.keep_uncertain_on_grid or dense_protected or visual_loose:
                kept.append(q)
            else:
                q["RejectReason"] = "grid_node_not_solder"
                removed.append(q)
        else:
            # 离所有主网格点：大多是过孔/器件/外围干扰。
            nearest_grid: Optional[GridModel] = None
            in_any_bbox = False
            for grid in grids:
                if (grid.bbox[0] <= cx <= grid.bbox[2]) and (grid.bbox[1] <= cy <= grid.bbox[3]):
                    nearest_grid = grid
                    in_any_bbox = True
                    break
            if nearest_grid is None:
                nearest_grid = grids[0]
            features = extract_node_features(image_bgr, cx, cy, nearest_grid.radius_ref, grid=nearest_grid, point=p, residual=None)
            score = score_node(features)
            visual_strong = _visual_solder_evidence(features, score, cfg, strong=True)
            visual_loose = _visual_solder_evidence(features, score, cfg, strong=False)
            q = dict(p)
            q["GridNodeLabel"] = score["label"]
            q["GridNodeSolderScore"] = round(float(score["solder_score"]), 4)
            q["GridNodeViaScore"] = round(float(score["via_score"]), 4)
            q["GridNodeReason"] = score["reason"]
            q["GridNodeVisualLoose"] = bool(visual_loose)
            q["GridNodeVisualStrong"] = bool(visual_strong)
            if cfg.attach_node_features:
                q["GridNodeFeatures"] = _round_feature_dict(features)

            keep_offgrid = (
                (cfg.keep_strong_solder_offgrid and in_any_bbox and score["solder_score"] >= cfg.min_add_score)
                or (cfg.keep_strong_visual_offgrid and visual_strong)
            )
            if cfg.remove_offgrid and not keep_offgrid:
                q["RejectReason"] = "off_all_main_grids"
                removed.append(q)
            else:
                kept.append(q)

    # 再补缺失节点：v3 强制视觉证据；强视觉证据可以跨大 gap，弱证据只允许小缺口。
    if cfg.add_missing:
        for grid in grids:
            gid = int(grid.grid_id)
            for row, col, gx, gy in iter_grid_nodes(grid):
                node = (gid, row, col)
                if node in node_to_idx:
                    continue
                if cfg.enable_void_mask and cfg.void_forbid_add and node in void_nodes:
                    continue

                rec = node_cache.get(node)
                if rec is not None:
                    ns = int(rec["neighbor_support"])
                    direct_ns = int(rec["direct_neighbor_support"])
                    features = rec["features"]
                    score = rec["score"]
                    visual_strong = bool(rec["visual_strong"])
                    visual_solid = bool(rec.get("visual_solid", False))
                else:
                    ns = _neighbor_support_multi(node_to_idx, gid, row, col)
                    direct_ns = _direct_neighbor_support_multi(node_to_idx, gid, row, col)
                    features = extract_node_features(
                        image_bgr, gx, gy, grid.radius_ref, grid=grid, point=None, residual=0.0, neighbor_support=ns
                    )
                    score = score_node(features)
                    visual_strong = _visual_solder_evidence(features, score, cfg, strong=True)
                    visual_solid = _solid_solder_evidence(features, score, cfg)

                if not _has_add_visual_evidence(features, score, cfg):
                    continue

                if not _boundary_add_allowed(grid, row, col, features, score, visual_strong, direct_ns, cfg):
                    continue
                if cfg.add_require_solid_evidence and not visual_solid:
                    continue

                gap_h = _missing_gap_len_bounded(node_to_idx, grid, row, col, axis="h")
                gap_v = _missing_gap_len_bounded(node_to_idx, grid, row, col, axis="v")
                small_gap_ok = min(gap_h, gap_v) <= cfg.max_add_gap_nodes and direct_ns >= cfg.add_min_direct_neighbor

                row_support, col_support = _line_visual_support(node_cache, grid, row, col)
                boundary_node = _is_boundary_margin_node(grid, row, col, cfg.forbid_add_boundary_margin)
                large_gap_ok = (
                    cfg.add_allow_large_gap_with_strong_visual
                    and visual_strong
                    and visual_solid
                    and row_support >= cfg.add_min_row_line_support
                    and col_support >= cfg.add_min_col_line_support
                    and not (cfg.disable_large_gap_add_on_boundary and boundary_node)
                )
                if not (small_gap_ok or large_gap_ok):
                    continue

                p_add = _make_added_point(gx, gy, grid.radius_ref, score, features)
                p_add["GridId"] = int(gid)
                p_add["GridRow"] = int(row)
                p_add["GridCol"] = int(col)
                p_add["GridGapH"] = int(gap_h)
                p_add["GridGapV"] = int(gap_v)
                p_add["GridRowSupport"] = int(row_support)
                p_add["GridColSupport"] = int(col_support)
                p_add["GridNodeVisualSolid"] = bool(visual_solid)
                p_add["GridBoundaryNode"] = bool(boundary_node)
                added.append(p_add)

    final_points = kept + added
    meta: Dict[str, Any] = {
        "enabled": True,
        "version": "v4_boundary_guard_solid_void",
        "grid_count": int(len(grids)),
        "pitch_x": round(float(np.median([g.pitch_x for g in grids])), 4),
        "pitch_y": round(float(np.median([g.pitch_y for g in grids])), 4),
        "radius_ref": round(float(np.median([g.radius_ref for g in grids])), 4),
        "grids": [
            {
                "grid_id": int(g.grid_id),
                "pitch_x": round(float(g.pitch_x), 4),
                "pitch_y": round(float(g.pitch_y), 4),
                "radius_ref": round(float(g.radius_ref), 4),
                "grid_rows": int(g.max_row - g.min_row + 1),
                "grid_cols": int(g.max_col - g.min_col + 1),
                "grid_node_count": int((g.max_row - g.min_row + 1) * (g.max_col - g.min_col + 1)),
                "main_component_count": int(len(g.main_indices)),
                "bbox": tuple(round(float(x), 3) for x in g.bbox),
            }
            for g in grids
        ],
        "assigned_count": int(len(idx_to_node)),
        "void_node_count": int(len(void_nodes)),
        "kept_count": int(len(kept)),
        "removed_count": int(len(removed)),
        "added_count": int(len(added)),
    }
    return final_points, removed, added, meta

def _round_feature_dict(features: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keep = [
        "radius_ratio", "grid_residual_norm", "neighbor_support",
        "core_mean", "fill_mean", "ring_mean", "outer_mean",
        "solder_fill_dark", "solder_core_dark", "via_center_score", "via_outer_score",
        "fill_dark_ratio", "core_bright_ratio", "ring_dark_ratio",
        "blob_valid", "blob_area_ratio", "blob_circularity", "blob_center_dist_norm",
        "blob_dark", "blob_fill_ratio", "blob_core_ratio",
        "circularity", "extent", "shape_aspect", "radial_cv", "contour_radius",
    ]
    for k in keep:
        if k in features:
            out[k] = round(float(features[k]), 4)
    return out


def apply_grid_node_refine_to_result_item(
    result_item: Dict[str, Any],
    cfg: Optional[GridNodeRefineConfig] = None,
    image_key: str = "image",
    points_key: str = "points",
    removed_key: str = "removed_points",
    added_key: str = "added_points",
) -> Dict[str, Any]:
    """直接作用在你当前 detector.py 的 result_item 上。

    会返回一个浅拷贝，不会原地修改原对象。
    """
    item = dict(result_item)
    image = item.get(image_key)
    points = list(item.get(points_key, []))

    refined, removed2, added2, meta = refine_points_by_grid_nodes(image, points, cfg=cfg)

    old_removed = list(item.get(removed_key, []))
    old_added = list(item.get(added_key, []))

    item[points_key] = refined
    item[removed_key] = old_removed + removed2
    item[added_key] = old_added + added2
    item["grid_node_meta"] = meta
    item["grid_node_removed_count"] = len(removed2)
    item["grid_node_added_count"] = len(added2)
    item["total_count"] = len(refined)
    return item


# -----------------------------
# 调试图
# -----------------------------

def draw_grid_node_debug(
    result_item: Dict[str, Any],
    show_removed: bool = True,
    show_label: bool = False,
) -> np.ndarray:
    """画第二阶段结果。

    颜色：
      - 绿色：保留点
      - 黄色：新增点
      - 红色：第二阶段删除点
    """
    image = result_item.get("image")
    if image is None:
        raise ValueError("result_item['image'] is required")
    vis = image.copy()

    for p in result_item.get("points", []):
        cx, cy = int(round(float(p["CenterX"]))), int(round(float(p["CenterY"])))
        r = int(round(float(p.get("Radius", 4))))
        if p.get("AddedByGridNode", False):
            color = (0, 255, 255)
            thickness = 2
        else:
            color = (0, 220, 0)
            thickness = 1
        cv2.circle(vis, (cx, cy), max(2, r), color, thickness)
        cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 7, 1)
        if show_label:
            txt = p.get("GridNodeLabel", "")[:1]
            cv2.putText(vis, txt, (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    if show_removed:
        for p in result_item.get("removed_points", []):
            if "RejectReason" not in p or not str(p["RejectReason"]).startswith(("grid_node", "off_main_grid", "off_all_main_grids")):
                continue
            cx, cy = int(round(float(p["CenterX"]))), int(round(float(p["CenterY"])))
            r = int(round(float(p.get("Radius", 4))))
            cv2.circle(vis, (cx, cy), max(2, r), (0, 0, 255), 1)
            cv2.line(vis, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 255), 1)
            cv2.line(vis, (cx - r, cy + r), (cx + r, cy - r), (0, 0, 255), 1)

    return vis


# -----------------------------
# CSV 特征导出：后面训练小分类器用
# -----------------------------

def export_grid_node_features_csv(
    image_bgr: np.ndarray,
    points: List[PointDict],
    csv_path: str,
    cfg: Optional[GridNodeRefineConfig] = None,
) -> Dict[str, Any]:
    """导出当前图片上的 grid-node 特征。

    CSV 里默认 label 为空。你可以人工把 label 填成 solder/via/background/component，
    后续用这些字段训练 RandomForest/LightGBM 替换 score_node()。
    v2 支持导出多个 grid。
    """
    import csv

    cfg = cfg or GridNodeRefineConfig()
    grids = fit_grid_models(points, cfg)
    if not grids:
        return {"ok": False, "reason": "grid_fit_failed"}

    node_to_idx, idx_to_node = assign_points_to_grid_models(points, grids, cfg)
    rows_out: List[Dict[str, Any]] = []

    for grid in grids:
        gid = int(grid.grid_id)
        for row, col, gx, gy in iter_grid_nodes(grid):
            idx = node_to_idx.get((gid, row, col))
            p = points[idx] if idx is not None else None
            residual = idx_to_node[idx][3] if idx is not None else 0.0
            ns = _neighbor_support_multi(node_to_idx, gid, row, col)
            direct_ns = _direct_neighbor_support_multi(node_to_idx, gid, row, col)
            features = extract_node_features(
                image_bgr, gx, gy, grid.radius_ref, grid=grid, point=p, residual=residual, neighbor_support=ns
            )
            score = score_node(features)
            rec: Dict[str, Any] = {
                "label": "",
                "grid_id": gid,
                "row": row,
                "col": col,
                "x": round(gx, 4),
                "y": round(gy, 4),
                "has_yolo_point": int(p is not None),
                "direct_neighbor_support": int(direct_ns),
                "rule_label": score["label"],
                "solder_score": round(float(score["solder_score"]), 4),
                "via_score": round(float(score["via_score"]), 4),
                "reason": score["reason"],
            }
            rec.update(_round_feature_dict(features))
            rows_out.append(rec)

    if not rows_out:
        return {"ok": False, "reason": "no_rows"}

    fieldnames = list(rows_out[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return {"ok": True, "rows": len(rows_out), "csv_path": csv_path, "grid_count": len(grids)}
