"""Temporal perceptual coherence using LPIPS between temporally-separated frames (**higher = closer / more coherent**)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from bench_common import default_results_path, load_benchmark_rows, read_selected_bgr, sample_frame_indices, save_standard_results
from torchmetric.lpips_metrics import LearnedPerceptualImagePatchSimilarityMetric


def compute_consistency_3d(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    gap = max(2, int(os.environ.get("CONSISTENCY3D_PAIR_GAP_FRAMES", "4")))
    max_frames = int(os.environ.get("CONSISTENCY3D_MAX_FRAMES", "32"))

    rows = load_benchmark_rows(json_dir)
    model_name = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")

    details: List[Dict[str, Any]] = []
    scores: List[float] = []

    lpips_metric = LearnedPerceptualImagePatchSimilarityMetric()

    for row in rows:
        for vp in row.get("video_list", []) or []:
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()

            dense = sample_frame_indices(max(total, 1), min(max_frames, max(total, 1)))
            frames = read_selected_bgr(vp, dense)
            if len(frames) <= gap:
                scores.append(float("nan"))
                details.append({"video_path": vp, "score": float("nan"), "note": "too_few_frames"})
                continue

            pals: List[float] = []
            for i in range(0, len(frames) - gap, gap):
                a, b = frames[i], frames[i + gap]
                lp = float(lpips_metric._compute_scores(a, b))
                pals.append(lp)
            mean_lpips = float(np.mean(pals)) if pals else float("nan")
            agg = float(np.exp(-mean_lpips)) if pals else float("nan")
            scores.append(agg if np.isfinite(agg) else 0.0)
            details.append({"video_path": vp, "score": agg, "mean_lpips_between_pairs": mean_lpips})

    avg = float(np.nanmean(scores)) if scores else 0.0
    outp = default_results_path(json_dir, model_name, dataset_json, "consistency_3d")
    save_standard_results(
        outp,
        avg_score=avg,
        summary_key="exp_neg_mean_lpips_temporal_gap",
        video_details=details,
        extra_summary={"pair_gap_estimator": gap},
    )
    print(f"[consistency_3d] Detailed results saved to {outp}")
    return avg, details
