"""
Camera translation trajectory error (**lower is better** → we report inverted score for runner convention).

Uses **paired trajectories supplied in benchmark JSON** (no SLAM bundled). Provide predicted and ground-truth
camera centers in ``auxiliary_info`` for each benchmark row:

- Preferred: ``auxiliary_info`` is a dict with::

    {
      \"translations_gt\": [[x,y,z], ...],   # shape (N,3)
      \"translations_pred\": [[x,y,z], ...],
    }

Or 4×4 poses::

    {\"poses_gt\": [...], \"poses_pred\": [...]}

Obtain trajectory estimates with your own toolchain, e.g. `DROID-SLAM <https://github.com/princeton-vl/DROID-SLAM>`_.
This repository does **not** ship third-party pose estimators.

The scalar returned to ``runner.py`` uses **score = exp(-mean_l2_aligned)** so that **higher = better**.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from bench_common import default_results_path, load_benchmark_rows, save_standard_results


def _as_translation_matrix_list(aux: Dict[str, Any], key_pairs: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    kp, ks = key_pairs
    gt = aux.get(kp)
    pr = aux.get(ks)
    if gt is None or pr is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    def from_poses(poses_any: Any) -> np.ndarray:
        arr = np.asarray(poses_any, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr[:, :3, 3]
        if arr.ndim == 3 and arr.shape[1:] == (3, 4):
            return arr[:, :3, 3]
        raise ValueError("poses must be (N,4,4) or (N,3,4)")
    if kp == "poses_gt":
        return from_poses(gt), from_poses(pr)
    return np.asarray(gt, dtype=np.float64).reshape(-1, 3), np.asarray(pr, dtype=np.float64).reshape(-1, 3)


def _scaled_l2(gt: np.ndarray, pr: np.ndarray) -> float:
    if gt.size == 0 or pr.size == 0:
        return float("nan")
    if gt.shape != pr.shape:
        raise ValueError(f"Trajectory length mismatch gt={gt.shape} pred={pr.shape}")
    denom = np.linalg.norm(gt.reshape(-1)) + 1e-12
    s = np.sum(gt * pr) / (np.sum(pr * pr) + 1e-12)
    err = gt - s * pr
    return float(np.mean(np.linalg.norm(err, axis=1)))


def compute_camera_transform_error(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows = load_benchmark_rows(json_dir)
    model_name = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")
    details: List[Dict[str, Any]] = []
    raw_errors: List[float] = []

    for row in rows:
        aux = row.get("auxiliary_info", {})
        if isinstance(aux, list):
            aux = aux[0] if aux and isinstance(aux[0], dict) else {}

        tl_gt = tl_pr = None
        try:
            if isinstance(aux, dict) and "translations_gt" in aux:
                tl_gt, tl_pr = _as_translation_matrix_list(aux, ("translations_gt", "translations_pred"))
            elif isinstance(aux, dict) and "poses_gt" in aux:
                tl_gt, tl_pr = _as_translation_matrix_list(aux, ("poses_gt", "poses_pred"))
        except Exception as e:
            for vp in row.get("video_list", []) or []:
                details.append({"video_path": vp, "error": str(e)})
            continue

        for vp in row.get("video_list", []) or []:
            if tl_gt is None:
                msg = (
                    "Missing translations_gt/translations_pred (or poses). "
                    "See module docstring; use external SLAM repos such as "
                    "https://github.com/princeton-vl/DROID-SLAM to produce trajectories."
                )
                details.append({"video_path": vp, "error": msg})
                continue
            l2 = _scaled_l2(tl_gt, tl_pr)
            if math.isnan(l2):
                details.append({"video_path": vp, "error": "nan_error"})
                continue
            raw_errors.append(l2)
            details.append({"video_path": vp, "mean_l2_scaled": l2, "score_inverse": math.exp(-l2)})

    avg_err = float(np.mean(raw_errors)) if raw_errors else float("inf")
    # Higher-is-better for runner display
    final = math.exp(-avg_err) if raw_errors else 0.0
    outp = default_results_path(json_dir, model_name, dataset_json, "camera_transform_error")
    save_standard_results(
        outp,
        avg_score=final,
        summary_key="exp_neg_mean_translation_error",
        video_details=details,
        extra_summary={"mean_l2_scaled": avg_err},
    )
    print(f"[camera_transform_error] Detailed results saved to {outp}")
    return final, details
