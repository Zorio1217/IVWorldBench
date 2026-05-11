"""VGG early-layer Gram statistics: penalise intra-video variance (higher aggregate = stabler «style»)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bench_common import default_results_path, load_benchmark_rows, read_selected_bgr, sample_frame_indices, save_standard_results


def gram_matrix(act: torch.Tensor) -> torch.Tensor:
    """act: [1, C, H, W]."""
    b, c, h, w = act.shape
    f = act.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


class _VGGStyleExtractor(nn.Module):
    """First sixteen VGG convolutional layers (classification head removed)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        feats = list(vgg16(weights=VGG16_Weights.DEFAULT).features[:16])
        self.net = nn.Sequential(*feats).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def compute_style_consistency(
    json_dir: str,
    device: str,
    submodules_dict: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    import torchvision.transforms as T

    rows = load_benchmark_rows(json_dir)
    model_name = kwargs.get("model", "")
    dataset_json = kwargs.get("dataset_json", "")
    max_frames = int(os.environ.get("STYLE_MAX_FRAMES", "24"))

    dev = torch.device(os.environ["BENCH_TORCH_DEVICE"])
    extractor = _VGGStyleExtractor().to(dev)

    preprocess = T.Compose([T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor()])
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
                scores.append(float("nan"))
                continue

            grams: List[torch.Tensor] = []
            for fr in frames:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                t = preprocess(rgb).unsqueeze(0).to(dev)
                act = extractor(t)
                grams.append(gram_matrix(act).squeeze(0))

            stacked = torch.stack(grams, dim=0)
            v = stacked.var(dim=0, unbiased=False).mean().item()

            agg = float(1.0 / (1.0 + v))
            scores.append(agg)
            details.append({"video_path": vp, "score": agg, "gram_mean_variance": float(v)})

    avg = float(np.nanmean(scores)) if scores else 0.0
    outp = default_results_path(json_dir, model_name, dataset_json, "style_consistency")
    save_standard_results(
        outp,
        avg_score=avg,
        summary_key="style_cohesion_score_inv_var",
        video_details=details,
    )
    print(f"[style_consistency] Detailed results saved to {outp}")
    return avg, details
