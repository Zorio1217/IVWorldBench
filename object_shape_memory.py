"""Global appearance persistence: cosine similarity between early & late pooled ResNet embeddings (higher = more stable)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from bench_common import default_results_path, load_benchmark_rows, read_selected_bgr, sample_frame_indices, save_standard_results


class _TinyEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        w = ResNet18_Weights.DEFAULT
        m = resnet18(weights=w)
        self.body = torch.nn.Sequential(*list(m.children())[:-1])
        self.body.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x).flatten(1)
        return F.normalize(z, dim=-1)


def _embed_video(embedder: _TinyEmbedder, frames_bgr: List[np.ndarray], dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    prep = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensors = []
    for fr in frames_bgr:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        tensors.append(prep(rgb).unsqueeze(0))
    stacked = torch.cat(tensors, dim=0).to(dev)
    z = embedder(stacked)
    head = z[: max(1, len(z) // 4)].mean(dim=0)
    tail = z[-max(1, len(z) // 4) :].mean(dim=0)
    return head, tail


def compute_object_shape_memory(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows = load_benchmark_rows(json_dir)
    model_name = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")
    dev = torch.device(os.environ.get("BENCH_TORCH_DEVICE") or device or "cuda:0")
    max_frames = int(os.environ.get("SHAPE_EMBED_MAX_FRAMES", "48"))

    emb = _TinyEmbedder().to(dev)

    scores: List[float] = []
    details: List[Dict[str, Any]] = []

    for row in rows:
        for vp in row.get("video_list", []) or []:
            cap = cv2.VideoCapture(vp)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
            idx = sample_frame_indices(max(total, 1), max_frames)
            frames = read_selected_bgr(vp, idx)
            if len(frames) < 4:
                details.append({"video_path": vp, "score": float("nan"), "note": "too_few_frames"})
                continue
            h, t = _embed_video(emb, frames, dev)
            sim = float((h * t).sum().item())
            scores.append(sim)
            details.append({"video_path": vp, "score": sim})

    avg = float(np.nanmean(scores)) if scores else 0.0
    outp = default_results_path(json_dir, model_name, dataset_json, "object_shape_memory")
    save_standard_results(outp, avg_score=avg, summary_key="mean_shape_similarity", video_details=details)
    print(f"[object_shape_memory] Detailed results saved to {outp}")
    return avg, details
