"""Shared loaders and Torch device bridging for perceptual benchmarks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch


def resolve_effective_device_token(pref: str) -> str:
    p = (pref or "").strip()
    lower = p.lower()
    if lower.startswith("cuda"):
        return p if torch.cuda.is_available() else "cpu"
    if lower == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return lower if lower in ("cpu", "mps") else "cpu"


def set_global_torch_device_for_metrics(token: str) -> None:
    os.environ["BENCH_TORCH_DEVICE"] = token


def load_benchmark_rows(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Benchmark intermediate JSON must be a list.")
    return data


def default_results_path(json_dir: str, model_name: str, dataset_json: str, stem: str) -> Path:
    out_dir = Path(json_dir).parent
    dj = Path(dataset_json).stem if dataset_json else "dataset"
    suffix = f"{stem}__{model_name}__{dj}_results.json" if model_name else f"{stem}_results.json"
    return out_dir / suffix


def save_standard_results(
    out_path: Path,
    *,
    avg_score: float,
    summary_key: str,
    video_details: List[Dict[str, Any]],
    extra_summary: Dict[str, Any] | None = None,
) -> None:
    summary: Dict[str, Any] = {"average_score": float(avg_score), summary_key: float(avg_score)}
    if extra_summary:
        summary.update(extra_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"evaluation_summary": summary, "video_details": video_details}, fh, indent=2, ensure_ascii=False)


def sample_frame_indices(total: int, max_frames: int) -> Sequence[int]:
    if total <= 0:
        return []
    cap = max(1, max_frames)
    if total <= cap:
        return list(range(total))
    return [int(v) for v in np.linspace(0, total - 1, num=cap, dtype=np.int64)]


def read_selected_bgr(video_path: str, indices: Sequence[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    buffers: List[np.ndarray] = []
    for idx in indices:
        if total > 0 and idx >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if ok:
            buffers.append(frame)
    cap.release()
    return buffers
