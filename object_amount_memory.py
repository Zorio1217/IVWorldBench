"""Object-count stability across frames (higher when counts are steadier vs frame index)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from bench_common import load_benchmark_rows, read_selected_bgr, sample_frame_indices, save_standard_results, default_results_path


def _counts_with_ultralytics(frames: List[np.ndarray], weights_path: str) -> List[int]:
    from ultralytics import YOLO

    model = YOLO(weights_path)
    out: List[int] = []
    for fr in frames:
        result = model.predict(fr, verbose=False, device=model.device)[0]
        n = len(result.boxes) if result.boxes is not None else 0
        out.append(int(n))
    return out


def _fallback_counts_edges(frames: List[np.ndarray]) -> List[int]:
    vals: List[int] = []
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 50, 150)
        vals.append(int(e.sum() // (e.shape[0] * e.shape[1] + 1)))
    return vals


def _stability_score(counts: List[float]) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    cv = float(arr.std(ddof=0) / (arr.mean() + 1e-6))
    return float(1.0 / (1.0 + cv))


def compute_object_amount_memory(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    import os

    rows = load_benchmark_rows(json_dir)
    model_name = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")
    weights = os.environ.get("YOLO_WEIGHTS", os.environ.get("ULTRALYTICS_WEIGHTS", "yolov8n.pt"))
    max_frames = int(os.environ.get("OBJECT_AMOUNT_MAX_FRAMES", "64"))

    details: List[Dict[str, Any]] = []
    scores: List[float] = []

    for row in rows:
        for vp in row.get("video_list", []) or []:
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            indices = sample_frame_indices(max(total, 1), max_frames)
            frames = read_selected_bgr(vp, indices)
            if len(frames) < 3:
                details.append({"video_path": vp, "score": 0.0, "note": "<3 frames"})
                scores.append(0.0)
                continue
            try:
                counts = _counts_with_ultralytics(frames, weights)
                backend = "ultralytics"
            except Exception as e:
                counts = _fallback_counts_edges(frames)
                backend = f"edge_fallback({e})"

            score = _stability_score([float(x) for x in counts])
            scores.append(score)
            details.append({"video_path": vp, "score": score, "counts": counts, "backend": backend})

    avg = float(np.nanmean(scores)) if scores else 0.0
    outp = default_results_path(json_dir, model_name, dataset_json, "object_amount_memory")
    save_standard_results(
        outp,
        avg_score=avg,
        summary_key="object_amount_stability",
        video_details=details,
        extra_summary={"weights": weights},
    )
    print(f"[object_amount_memory] Detailed results saved to {outp}")
    return avg, details
