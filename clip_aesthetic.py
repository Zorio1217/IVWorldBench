"""CLIP text–image similarity with aesthetic prose (TorchMetrics ``CLIPScore``)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as VF

from bench_common import default_results_path, load_benchmark_rows, read_selected_bgr, sample_frame_indices, save_standard_results

_DEFAULT_PROMPT = "a cinematic, beautiful, aesthetically pleasing masterpiece"


def compute_clip_aesthetic(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    from torchmetrics.multimodal.clip_score import CLIPScore

    caption_prompt = os.environ.get("CLIP_AESTHETIC_PROMPT", _DEFAULT_PROMPT).strip()
    captions = [caption_prompt]

    rows = load_benchmark_rows(json_dir)
    label = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")
    max_frames = int(os.environ.get("CLIP_AESTHETIC_MAX_FRAMES", "12"))

    scorer = CLIPScore().to(torch.device(os.environ["BENCH_TORCH_DEVICE"]))
    scores: List[float] = []
    details: List[Dict[str, Any]] = []

    for row in rows:
        for vp in row.get("video_list", []) or []:
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            idx = sample_frame_indices(max(total, 1), max_frames)
            imgs = read_selected_bgr(vp, idx)

            vals: List[float] = []
            for fr in imgs:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                x = VF.to_tensor(rgb).unsqueeze(0).clamp(0, 1).to(torch.device(os.environ["BENCH_TORCH_DEVICE"]))
                with torch.no_grad():
                    out = scorer(x, captions)
                vals.append(float(out.squeeze().clamp(-999, 999).cpu().detach().numpy()))
            agg = float(np.mean(vals)) if vals else float("nan")
            scores.append(0.0 if np.isnan(agg) else agg)
            details.append({"video_path": vp, "mean_clip_alignment": agg, "captions_used": captions})

    overall = float(np.nanmean(scores)) if scores else 0.0
    outp = default_results_path(json_dir, label, dataset_json, "clip_aesthetic")
    save_standard_results(outp, avg_score=overall, summary_key="mean_clip_alignment", video_details=details)
    print(f"[clip_aesthetic] Detailed results saved to {outp}")
    return overall, details
