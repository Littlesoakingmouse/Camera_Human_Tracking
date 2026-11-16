"""Tracking metrics: CLEAR MOT, identity (IDF1) and HOTA.

Implemented with numpy/scipy only, following the reference definitions:
- CLEAR MOT: Bernardin & Stiefelhagen 2008 (as in the MOTChallenge devkit).
- IDF1: Ristani et al. 2016 (global bipartite matching of identities).
- HOTA: Luiten et al. 2021, same algorithm as TrackEval (alpha 0.05..0.95).

Every function takes two DataFrames with columns ``frame, id, x, y, w, h``.
``frame`` may be any hashable-free integer key; multi-camera evaluation uses
``camera_index * 10**7 + frame`` so one identity can span several cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLD = 0.5
EPS = np.finfo(float).eps


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between ``(N, 4)`` and ``(M, 4)`` arrays in x, y, w, h format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, EPS), 0.0)


@dataclass
class _Frames:
    """GT/prediction boxes grouped per frame with ids remapped to 0..K-1."""

    keys: list
    gt_ids: list[np.ndarray]
    gt_boxes: list[np.ndarray]
    pr_ids: list[np.ndarray]
    pr_boxes: list[np.ndarray]
    num_gt_ids: int
    num_pr_ids: int


def _group(gt: pd.DataFrame, pred: pd.DataFrame) -> _Frames:
    gt_codes, gt_uniques = pd.factorize(gt["id"])
    pr_codes, pr_uniques = pd.factorize(pred["id"])
    gt = gt.assign(_id=gt_codes)
    pred = pred.assign(_id=pr_codes)
    gt_groups = {k: g for k, g in gt.groupby("frame", sort=False)}
    pr_groups = {k: g for k, g in pred.groupby("frame", sort=False)}
    keys = sorted(set(gt_groups) | set(pr_groups))
    empty_ids, empty_boxes = np.zeros(0, int), np.zeros((0, 4))
    frames = _Frames(keys, [], [], [], [], len(gt_uniques), len(pr_uniques))
    for key in keys:
        g, p = gt_groups.get(key), pr_groups.get(key)
        frames.gt_ids.append(g["_id"].to_numpy() if g is not None else empty_ids)
        frames.gt_boxes.append(g[["x", "y", "w", "h"]].to_numpy(float) if g is not None else empty_boxes)
        frames.pr_ids.append(p["_id"].to_numpy() if p is not None else empty_ids)
        frames.pr_boxes.append(p[["x", "y", "w", "h"]].to_numpy(float) if p is not None else empty_boxes)
    return frames


def remove_ignored(pred: pd.DataFrame, gt_all: pd.DataFrame) -> pd.DataFrame:
    """Drop predictions matched to ignore regions (MOT17-style preprocessing).

    Predictions are matched by Hungarian IoU >= 0.5 against all GT boxes of the
    frame (persons + ignore regions); those assigned to an ignore box are removed.
    ``gt_all`` must contain a boolean ``ignore`` column.
    """
    if not gt_all["ignore"].any() or pred.empty:
        return pred
    keep = np.ones(len(pred), dtype=bool)
    pred_positions = {k: idx for k, idx in pred.groupby("frame", sort=False).indices.items()}
    for frame, gt_frame in gt_all.groupby("frame", sort=False):
        if not gt_frame["ignore"].any() or frame not in pred_positions:
            continue
        rows = pred_positions[frame]
        ious = iou_matrix(gt_frame[["x", "y", "w", "h"]].to_numpy(float),
                          pred.iloc[rows][["x", "y", "w", "h"]].to_numpy(float))
        score = np.where(ious >= IOU_THRESHOLD, ious, 0.0)
        gi, pi = linear_sum_assignment(-score)
        ignore_flags = gt_frame["ignore"].to_numpy()
        for g, p in zip(gi, pi):
            if score[g, p] > 0 and ignore_flags[g]:
                keep[rows[p]] = False
    return pred[keep]


def clear_mot(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    f = _group(gt, pred)
    tp = fp = fn = idsw = 0
    iou_sum = 0.0
    last_match = np.full(f.num_gt_ids, -1)  # last matched prediction id per GT id
    tracked_prev = np.zeros(f.num_gt_ids, dtype=bool)
    frag = np.zeros(f.num_gt_ids, dtype=int)
    ever_tracked = np.zeros(f.num_gt_ids, dtype=bool)
    gt_frames = np.zeros(f.num_gt_ids, dtype=int)
    gt_tracked_frames = np.zeros(f.num_gt_ids, dtype=int)
    prev_pairs: dict[int, int] = {}
    for g_ids, g_box, p_ids, p_box in zip(f.gt_ids, f.gt_boxes, f.pr_ids, f.pr_boxes):
        ious = iou_matrix(g_box, p_box)
        score = np.where(ious >= IOU_THRESHOLD - EPS, ious, 0.0)
        # Keep last frame's correspondences when still valid (MOTChallenge rule).
        for gi, gid in enumerate(g_ids):
            if gid in prev_pairs:
                matches = np.where(p_ids == prev_pairs[gid])[0]
                if len(matches) and score[gi, matches[0]] > 0:
                    score[gi, matches[0]] += 1000.0
        rows, cols = linear_sum_assignment(-score)
        valid = score[rows, cols] > 0
        rows, cols = rows[valid], cols[valid]
        current_pairs = {}
        matched_gt = np.zeros(len(g_ids), dtype=bool)
        for r, c in zip(rows, cols):
            gid, pid = int(g_ids[r]), int(p_ids[c])
            if last_match[gid] != -1 and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
            current_pairs[gid] = pid
            iou_sum += ious[r, c]
            matched_gt[r] = True
        tp += len(rows)
        fn += len(g_ids) - len(rows)
        fp += len(p_ids) - len(rows)
        gt_frames[g_ids] += 1
        gt_tracked_frames[g_ids[matched_gt]] += 1
        now_tracked = np.zeros(f.num_gt_ids, dtype=bool)
        now_tracked[g_ids[matched_gt]] = True
        present = np.zeros(f.num_gt_ids, dtype=bool)
        present[g_ids] = True
        # A fragmentation is a GT track that resumes tracking after an interruption.
        resumed = now_tracked & ~tracked_prev & ever_tracked
        frag += resumed
        ever_tracked |= now_tracked
        # Frames where a GT id is not annotated do not interrupt its tracking state.
        tracked_prev = np.where(present, now_tracked, tracked_prev)
        prev_pairs = current_pairs
    num_gt = tp + fn
    ratio = np.divide(gt_tracked_frames, np.maximum(gt_frames, 1))
    return {
        "MOTA": 1.0 - (fn + fp + idsw) / max(num_gt, 1),
        "MOTP": iou_sum / max(tp, 1),
        "Precision": tp / max(tp + fp, 1),
        "Recall": tp / max(num_gt, 1),
        "TP": tp, "FP": fp, "FN": fn, "IDSW": idsw, "Frag": int(frag.sum()),
        "MT": int((ratio >= 0.8).sum()), "ML": int((ratio < 0.2).sum()),
        "GT_IDs": f.num_gt_ids, "Pred_IDs": f.num_pr_ids,
    }


def identity_metrics(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    f = _group(gt, pred)
    counts = np.zeros((f.num_gt_ids, f.num_pr_ids))
    for g_ids, g_box, p_ids, p_box in zip(f.gt_ids, f.gt_boxes, f.pr_ids, f.pr_boxes):
        if len(g_ids) and len(p_ids):
            hit = iou_matrix(g_box, p_box) >= IOU_THRESHOLD - EPS
            gi, pi = np.nonzero(hit)
            np.add.at(counts, (g_ids[gi], p_ids[pi]), 1)
    num_gt, num_pr = len(gt), len(pred)
    idtp = 0.0
    if counts.size:
        rows, cols = linear_sum_assignment(-counts)
        idtp = float(counts[rows, cols].sum())
    idfn, idfp = num_gt - idtp, num_pr - idtp
    return {
        "IDF1": 2 * idtp / max(2 * idtp + idfp + idfn, 1),
        "IDP": idtp / max(num_pr, 1),
        "IDR": idtp / max(num_gt, 1),
        "IDTP": idtp, "IDFP": idfp, "IDFN": idfn,
    }


def hota(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    f = _group(gt, pred)
    alphas = np.arange(0.05, 0.99, 0.05)
    potential = np.zeros((f.num_gt_ids, f.num_pr_ids))
    gt_count = np.zeros((f.num_gt_ids, 1))
    pr_count = np.zeros((1, f.num_pr_ids))
    sims = []
    for g_ids, g_box, p_ids, p_box in zip(f.gt_ids, f.gt_boxes, f.pr_ids, f.pr_boxes):
        sim = iou_matrix(g_box, p_box)
        sims.append(sim)
        if sim.size:
            denom = sim.sum(0)[None] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(denom > EPS, sim / np.maximum(denom, EPS), 0.0)
            potential[g_ids[:, None], p_ids[None]] += sim_iou
        gt_count[g_ids] += 1
        pr_count[0, p_ids] += 1
    global_score = potential / np.maximum(gt_count + pr_count - potential, EPS)
    tp, fn, fp, loc = (np.zeros(len(alphas)) for _ in range(4))
    match_counts = np.zeros((len(alphas), f.num_gt_ids, f.num_pr_ids))
    for g_ids, p_ids, sim in zip(f.gt_ids, f.pr_ids, sims):
        if len(g_ids) == 0 or len(p_ids) == 0:
            fn += len(g_ids)
            fp += len(p_ids)
            continue
        score = global_score[g_ids[:, None], p_ids[None]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(alphas):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            tp[a] += len(r)
            fn[a] += len(g_ids) - len(r)
            fp[a] += len(p_ids) - len(r)
            loc[a] += sim[r, c].sum()
            match_counts[a, g_ids[r], p_ids[c]] += 1
    ass_a = np.zeros(len(alphas))
    for a in range(len(alphas)):
        mc = match_counts[a]
        ass = mc / np.maximum(gt_count + pr_count - mc, 1)
        ass_a[a] = (mc * ass).sum() / max(tp[a], 1)
    det_a = tp / np.maximum(tp + fn + fp, 1)
    hota_alpha = np.sqrt(det_a * ass_a)
    return {
        "HOTA": float(hota_alpha.mean()),
        "DetA": float(det_a.mean()),
        "AssA": float(ass_a.mean()),
        "LocA": float((loc / np.maximum(tp, 1)).mean()),
    }


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    """All metrics for one sequence (or one merged multi-camera sequence)."""
    gt = gt[["frame", "id", "x", "y", "w", "h"]]
    pred = pred[["frame", "id", "x", "y", "w", "h"]]
    result: dict[str, float] = {}
    result.update(hota(gt, pred))
    result.update(clear_mot(gt, pred))
    result.update(identity_metrics(gt, pred))
    return result
