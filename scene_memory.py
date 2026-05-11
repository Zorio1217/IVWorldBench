"""Scene / long-horizon memory via symmetric-frame PSNR and LPIPS (higher is better)."""

from __future__ import annotations

import cv2
from typing import Any, Dict, List, Tuple

import numpy as np

from bench_common import default_results_path, load_benchmark_rows, save_standard_results
from torchmetric.lpips_metrics import LearnedPerceptualImagePatchSimilarityMetric
from torchmetric.psnr_metrics import PeakSignalNoiseRatioMetric


def _read_all_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def _symmetric_pairs(frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(frames)
    return [(frames[i], frames[n - 1 - i]) for i in range(n)]


def compute_scene_memory(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows = load_benchmark_rows(json_dir)
    model_name = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")

    details: List[Dict[str, Any]] = []
    scores: List[float] = []

    for row in rows:
        for vp in row.get("video_list", []) or []:
            caps = _read_all_frames(vp)
            pairs = _symmetric_pairs(caps)
            if not pairs:
                details.append({"video_path": vp, "score": float("nan"), "error": "no_frames"})
                continue
            lpips_metric = LearnedPerceptualImagePatchSimilarityMetric()
            psnr_metric = PeakSignalNoiseRatioMetric()
            psnrs: List[float] = []
            lpips_scores: List[float] = []
            for a, b in pairs:
                psnrs.append(float(psnr_metric._compute_scores(a, b)))
                lpips_scores.append(float(lpips_metric._compute_scores(a, b)))
            psnr_m = float(np.clip(np.mean(psnrs), 0.0, 100.0))
            lpips_m = float(np.mean(lpips_scores))
            s = float(0.5 * (psnr_m / 50.0) + 0.5 * (1.0 - min(lpips_m, 2.0) / 2.0))
            scores.append(s)
            details.append(
                {
                    "video_path": vp,
                    "score": s,
                    "mean_psnr": psnr_m,
                    "mean_lpips": lpips_m,
                    "num_pairs": len(pairs),
                }
            )

    final = float(np.nanmean(scores)) if scores else 0.0
    out_path = default_results_path(json_dir, model_name, dataset_json, "scene_memory")
    save_standard_results(
        out_path,
        avg_score=final,
        summary_key="mean_scene_memory_aggregate",
        video_details=details,
    )
    print(f"[scene_memory] Detailed results saved to {out_path}")
    return final, details
