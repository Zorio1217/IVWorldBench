"""CLIP-based no-reference perceptual score via PyIQA (CLIP-IQA family). Higher typically means stronger perceptual fidelity."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pyiqa
import torch
import torchvision.transforms.functional as VF

from bench_common import default_results_path, load_benchmark_rows, read_selected_bgr, sample_frame_indices, save_standard_results


def _pick_metric() -> Tuple[torch.nn.Module, str]:
    dev = torch.device(os.environ["BENCH_TORCH_DEVICE"])
    preferred = os.environ.get("CLIP_IQA_PYIQA_METRIC") or ""
    candidates = []
    if preferred.strip():
        candidates.append(preferred.strip())
    candidates += ["clipiqa+_vitL14_instruct", "clipiqa+", "clipiqa+_vitB16"]
    last_err: Exception | None = None
    for name in candidates:
        try:
            m = pyiqa.create_metric(name, device=dev)
            return m.to(dev), name
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Unable to initialise a PyIQA CLIP-IQA metric "
        "(candidates {}). Install weights / try CLIP_IQA_PYIQA_METRIC. Final error: {}".format(
            candidates, last_err
        )
    )


def compute_clip_iqa(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    metric, metric_backend = _pick_metric()
    dev = torch.device(os.environ["BENCH_TORCH_DEVICE"])

    rows = load_benchmark_rows(json_dir)
    label = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")
    max_frames = int(os.environ.get("CLIP_IQA_MAX_FRAMES", "16"))

    details: List[Dict[str, Any]] = []
    scores: List[float] = []

    for row in rows:
        for vp in row.get("video_list", []) or []:
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            idx = sample_frame_indices(max(total, 1), max_frames)
            imgs_bgr = read_selected_bgr(vp, idx)

            vals: List[float] = []
            for fr in imgs_bgr:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                tens = VF.to_tensor(rgb).unsqueeze(0).clamp(0, 1).to(dev)

                score_t = metric(tens)
                vals.append(float(score_t.squeeze().detach().cpu().clamp(-1e6, 1e6).item()))

            agg = float(np.mean(vals)) if vals else float("nan")
            scores.append(0.0 if np.isnan(agg) else agg)
            details.append({"video_path": vp, "mean_nr_metric": agg, "pyiqa_metric": metric_backend})

    overall = float(np.nanmean(scores)) if scores else 0.0
    outp = default_results_path(json_dir, label, dataset_json, "clip_iqa")
    save_standard_results(
        outp,
        avg_score=overall,
        summary_key="mean_clip_nr_metric",
        video_details=details,
        extra_summary={"backend": metric_backend},
    )
    print(f"[clip_iqa] Detailed results saved to {outp}")
    return overall, details
