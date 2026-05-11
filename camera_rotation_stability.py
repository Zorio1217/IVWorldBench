"""Rotation smoothness proxy from consecutive Essential-matrix estimates (**higher = smoother**)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from bench_common import default_results_path, load_benchmark_rows, read_selected_bgr, sample_frame_indices, save_standard_results


def _rot_angle_deg(rvec: np.ndarray) -> float:
    theta = np.linalg.norm(rvec)
    if theta < 1e-10:
        return 0.0
    return float(np.rad2deg(theta))


def _intrinsic_from_hw(h: int, w: int) -> np.ndarray:
    fx = float(max(w, 1))
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _pairwise_rotation_magnitudes(frames_gray: Sequence[np.ndarray]) -> List[float]:
    if len(frames_gray) < 2:
        return []
    orb = cv2.ORB_create(1200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    deltas: List[float] = []
    for a, b in zip(frames_gray[:-1], frames_gray[1:]):
        kp1, desc1 = orb.detectAndCompute(a, None)
        kp2, desc2 = orb.detectAndCompute(b, None)
        if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
            continue
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        if len(matches) < 8:
            continue
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        h, w = a.shape[:2]
        intr = _intrinsic_from_hw(h, w)

        essential, mask = cv2.findEssentialMat(
            pts1, pts2, intr, method=cv2.RANSAC, prob=0.999, threshold=3.0
        )
        if essential is None or essential.shape != (3, 3):
            continue
        _, R, _, _ = cv2.recoverPose(essential, pts1, pts2, intr, mask)
        rv, _ = cv2.Rodrigues(R)
        deltas.append(_rot_angle_deg(rv))

    return deltas


def compute_camera_rotation_stability(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows = load_benchmark_rows(json_dir)
    model_name = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")
    max_frames = 40

    details: List[Dict[str, Any]] = []
    scores: List[float] = []

    for row in rows:
        for vp in row.get("video_list", []) or []:
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            idx = sample_frame_indices(max(total, 1), max_frames)
            imgs = read_selected_bgr(vp, idx)
            gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
            d = _pairwise_rotation_magnitudes(gray)
            if len(d) < 2:
                s = float("nan")
            else:
                cv = float(np.std(d))
                mean = float(np.mean(d))
                s = float(1.0 / (1.0 + cv + mean * 0.01))
            scores.append(s if not np.isnan(s) else 0.0)
            details.append({"video_path": vp, "score": 0.0 if np.isnan(s) else s, "rots_deg": d})

    final = float(np.nanmean([x for x in scores if np.isfinite(x)])) if scores else 0.0
    outp = default_results_path(json_dir, model_name, dataset_json, "camera_rotation_stability")
    save_standard_results(
        outp,
        avg_score=final,
        summary_key="rotation_stability_score",
        video_details=details,
    )
    print(f"[camera_rotation_stability] Detailed results saved to {outp}")
    return final, details
