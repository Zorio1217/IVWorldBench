#!/usr/bin/env python3
"""Benchmark runner — nine perceptual / camera proxies only."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench_common import resolve_effective_device_token, set_global_torch_device_for_metrics

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".gif")

ALLOWED_DIMENSIONS: tuple[str, ...] = (
    "scene_memory",
    "object_amount_memory",
    "object_shape_memory",
    "camera_transform_error",
    "camera_rotation_stability",
    "clip_iqa",
    "clip_aesthetic",
    "consistency_3d",
    "style_consistency",
)


def _normalize_dim_name(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_").lower()


def discover_dimensions() -> List[str]:
    stems: List[str] = []
    for stem in ALLOWED_DIMENSIONS:
        if (ROOT / f"{stem}.py").is_file():
            stems.append(stem)
    return stems


def read_dataset_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)

    base_dir = ""
    if isinstance(root, dict):
        ds_base_path = root.get("dataset_info", {}).get("base_path", "")
        if ds_base_path:
            abs_path = os.path.abspath(path)
            idx = abs_path.find(ds_base_path)
            if idx > 0:
                base_dir = abs_path[:idx]

    items: List[Dict[str, Any]] = []
    models = root.get("models", []) if isinstance(root, dict) else []
    for m in models:
        model_name = m.get("model_name")
        for cond in m.get("conditions", []) or []:
            dim_raw = cond.get("condition_meta_info", "")
            dim_norm = _normalize_dim_name(dim_raw) if dim_raw else ""
            for p in cond.get("prompts", []) or []:
                videos = p.get("generated_videos", []) or []
                if base_dir:
                    videos = [
                        os.path.join(base_dir, v) if not os.path.isabs(v) else v for v in videos
                    ]
                prompt_text = p.get("condition_caption") or ""
                items.append(
                    {
                        "model": model_name,
                        "dimension": dim_norm,
                        "video_list": videos,
                        "prompt": prompt_text,
                        "auxiliary_info": p.get("auxiliary_info", []),
                    }
                )
    return items


def build_model_items(items: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    return [it for it in items if (not model_name or it.get("model") == model_name)]


def collect_videos_under_dir(path: str, recursive: bool) -> List[str]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return []
    files: List[Path] = []
    globber = root.rglob("*") if recursive else root.glob("*")
    for p in globber:
        if not p.is_file():
            continue
        if p.suffix.lower() in VIDEO_SUFFIXES:
            files.append(p)
    files.sort(key=lambda x: str(x))
    return [str(x) for x in files]


def create_single_video_item(model_name: str, video_path: str, prompt: str = "") -> Dict[str, Any]:
    return {
        "model": model_name,
        "dimension": "single_video",
        "video_list": [video_path],
        "prompt": prompt or "",
        "auxiliary_info": [],
    }


def import_dimension_module(dimension: str):
    if dimension not in ALLOWED_DIMENSIONS:
        sys.exit(f"Unsupported dimension '{dimension}'. Choose one of: {', '.join(ALLOWED_DIMENSIONS)}")
    return importlib.import_module(dimension)


def resolve_compute_fn(mod, dimension: str):
    norm = _normalize_dim_name(dimension)
    preferred = f"compute_{norm}"
    if hasattr(mod, preferred) and callable(getattr(mod, preferred)):
        return getattr(mod, preferred)
    names = sorted(n for n in dir(mod) if n.startswith("compute_") and callable(getattr(mod, n)))
    if not names:
        raise AttributeError(f"No compute_* export in '{dimension}'.")
    if len(names) == 1:
        return getattr(mod, names[0])
    hits = [n for n in names if norm.replace("_", "") == n[len("compute_") :].replace("_", "")]
    if len(hits) == 1:
        return getattr(mod, hits[0])
    raise AttributeError(f"Ambiguous exports {names}; implement compute_{norm} explicitly.")


def run_dimension(dimension: str, json_path: str, torch_device_token: str, *, model_name: str, dataset_json: str):
    mod = import_dimension_module(dimension)
    compute_fn = resolve_compute_fn(mod, dimension)
    score, details = compute_fn(json_path, torch_device_token, {}, model=model_name, dataset_json=dataset_json or "")
    return {"dimension": dimension, "score": float(score), "details": details}


def write_dimension_json(json_dir_root: str, dimension: str, model_items: List[Dict[str, Any]]) -> str:
    os.makedirs(json_dir_root, exist_ok=True)
    out_items: List[Dict[str, Any]] = []
    for it in model_items:
        out_items.append(
            {
                "dimension": [dimension],
                "video_list": it.get("video_list", []),
                "prompt_en": it.get("prompt", ""),
                "auxiliary_info": it.get("auxiliary_info", []),
            }
        )
    path = os.path.join(json_dir_root, f"{dimension}.json")
    with open(path, "w", encoding="utf-8") as wf:
        json.dump(out_items, wf, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight perceptual benchmark (nine dimensions).")
    parser.add_argument(
        "--dimension",
        required=False,
        help=f"One of: {', '.join(ALLOWED_DIMENSIONS)}",
    )
    parser.add_argument("--model_name", "--model", dest="model_name", default=None)

    inp = parser.add_mutually_exclusive_group(required=False)
    inp.add_argument("--video_path")
    inp.add_argument("--dataset_path")
    inp.add_argument("--dataset_json")

    parser.add_argument("--prompt", default="")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--json_out_dir", default=os.environ.get("BENCH_JSON_OUT", "dimension_description_json"))
    parser.add_argument("--results_dir", default=os.environ.get("BENCH_RESULTS_DIR", "results"))

    parser.add_argument("--combine_videos", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--list_dimensions", action="store_true")

    args = parser.parse_args()

    if args.list_dimensions:
        found = discover_dimensions()
        if not found:
            print("Missing metric modules next to runner.py.", file=sys.stderr)
        else:
            for stem in found:
                print(stem)
        return

    if not args.dimension:
        parser.error("--dimension required unless listing.")
    dimension = args.dimension.strip()
    if dimension not in ALLOWED_DIMENSIONS:
        parser.error(f"--dimension must be one of {list(ALLOWED_DIMENSIONS)}")
    if not args.model_name:
        parser.error("--model_name is required.")

    resolved = resolve_effective_device_token(args.device)
    if resolved == "cpu" and str(args.device).lower().startswith("cuda"):
        print(
            "[runner] CUDA requested via --device but this PyTorch build has no CUDA; falling back to CPU.\n"
            "Install a CUDA-capable Torch build on GPU machines.",
            file=sys.stderr,
        )

    dataset_json_used: Optional[str] = None
    model_items: List[Dict[str, Any]]
    slug = args.model_name

    if args.video_path:
        v = Path(args.video_path).expanduser()
        if not v.is_file():
            sys.exit(f"Video missing: {v}")
        model_items = [create_single_video_item(args.model_name, str(v.resolve()), args.prompt)]
        slug = f"{slug}_{v.stem}"
    elif args.dataset_path:
        dpath = Path(args.dataset_path).expanduser()
        if not dpath.is_dir():
            sys.exit(f"Not a folder: {dpath}")
        vids = collect_videos_under_dir(str(dpath), args.recursive)
        if not vids:
            sys.exit("No usable videos.")
        if args.combine_videos:
            model_items = [
                {
                    "model": args.model_name,
                    "dimension": "batch",
                    "video_list": vids,
                    "prompt": args.prompt,
                    "auxiliary_info": [],
                }
            ]
        else:
            model_items = [create_single_video_item(args.model_name, vp, args.prompt) for vp in vids]
        slug = f"{slug}_{dpath.name}_n{len(vids)}"
    elif args.dataset_json:
        dataset_json_used = str(Path(args.dataset_json).expanduser().resolve())
        bundle = build_model_items(read_dataset_json(dataset_json_used), args.model_name)
        if not bundle:
            sys.exit("dataset_json yielded zero rows after model filtering.")
        if args.prompt:
            for piece in bundle:
                piece["prompt"] = args.prompt
        model_items = bundle
        slug += f"_{Path(dataset_json_used).stem}"
    else:
        parser.error("Provide --video_path, --dataset_path, or --dataset_json")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    dim_json_path = write_dimension_json(args.json_out_dir, dimension, model_items)

    torch_token = resolved
    if torch_token.lower() == "cuda":
        torch_token = "cuda:0"
    set_global_torch_device_for_metrics(torch_token)

    print(f"[runner] {dimension} · model={args.model_name} · Torch device env={torch_token} · json={dim_json_path}")

    tic = time.time()
    outcome = run_dimension(dimension, dim_json_path, torch_token, model_name=args.model_name, dataset_json=dataset_json_used or "")
    outcome["elapsed_sec"] = time.time() - tic
    npz_target = Path(args.results_dir) / f"{slug}__{dimension}.npz"
    np.savez(npz_target, score=outcome["score"], details=outcome["details"], elapsed_sec=outcome["elapsed_sec"])
    print("\n=== Summary ===")
    print(f"score={outcome['score']:.5f}\ttime={outcome['elapsed_sec']:.2f}s\nsaved_npz={npz_target}")


if __name__ == "__main__":
    main()
