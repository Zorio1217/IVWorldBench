#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""计算指定结果 JSON 中每个 dimension 的平均分, 并写回到 evaluation_summary 中。

用法:
  python compute_dimension_averages.py path/to/results1.json [path/to/results2.json ...]
若不传参, 默认使用 JSON_PATHS 列表中的路径进行批处理

输出:
  对每个文件:
    1) 在同目录生成原文件的备份(第一次加 .bak, 若已存在递增编号)
    2) 更新 evaluation_summary:
          {
            "total_videos": ...,
            "total_score": ...,
            "average_score": ...,
            "per_dimension": {
                "dynamics": {"count": n, "total_score": x, "average_score": a},
                ...
            }
          }
    按顺序把 per_dimension 放在 average_score 之后。
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from collections import OrderedDict, defaultdict

DEFAULT_PATH = [
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_vista__video-to-4D-physical_results_v5.json"),
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_traj__video-to-4D-physical_results_v5.json"),
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_recam__video-to-4D-physical_results_v5.json"),
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_ex4d__video-to-4D-physical_results_v5.json"),
    Path("dimension_description_json/physics_reasoning__Diffusion_as_Shader__image-any-physical-image-to-4D_results_v5_dim.json"),
    Path("dimension_description_json/physics_reasoning__CamI2V__image-any-physical-image-to-4D_results_v5_dim.json"),
    Path("dimension_description_json/physics_reasoning__dream_in_4D__text_to_4D_physical_results_v5.json"),
    Path("dimension_description_json/physics_reasoning__4d-fy__text_to_4D_physical_results_v5.json"),
]
# 新增: 默认批处理列表(可自行增减)
# JSON_PATHS 需为扁平的 Path 列表，避免出现嵌套列表
JSON_PATHS: list[Path] = DEFAULT_PATH


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def backup_file(path: Path) -> Path:
    """创建不覆盖的备份, 返回备份路径"""
    base_backup = path.with_suffix(path.suffix + ".bak")
    if not base_backup.exists():
        path.replace(base_backup)
        return base_backup
    # 递增编号
    i = 1
    while True:
        alt = path.with_suffix(path.suffix + f".bak{i}")
        if not alt.exists():
            path.replace(alt)
            return alt
        i += 1


def compute_dimension_stats(video_details: list[dict]) -> dict:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for item in video_details:
        dim = item.get("dimension")
        score = item.get("score")
        if dim is None:
            continue
        # 仅接受数值型 score
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            continue
        sums[dim] += score_val
        counts[dim] += 1

    stats = {}
    for dim, cnt in counts.items():
        total = sums[dim]
        avg = (total / cnt) if cnt else 0.0
        stats[dim] = {
            "count": cnt,
            "total_score": round(total, 6),
            "average_score": round(avg, 6),
        }
    return stats


def recompute_overall(video_details: list[dict]) -> tuple[int, float, float]:
    total_videos = len(video_details)
    total_score = 0.0
    valid_scores = 0
    for item in video_details:
        try:
            s = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        total_score += s
        valid_scores += 1
    avg_score = total_score / valid_scores if valid_scores else 0.0
    return total_videos, total_score, avg_score


def update_evaluation_summary(data: dict) -> bool:
    video_details = data.get("video_details", [])
    if not isinstance(video_details, list):
        raise ValueError("JSON 格式错误: video_details 不是列表")

    per_dim = compute_dimension_stats(video_details)
    total_videos, total_score, avg_score = recompute_overall(video_details)

    # 构建有序 dict, 保证 per_dimension 排在 average_score 后
    new_summary = OrderedDict()
    new_summary["total_videos"] = total_videos
    new_summary["total_score"] = round(total_score, 6)
    new_summary["average_score"] = round(avg_score, 6)
    new_summary["per_dimension"] = per_dim

    old_summary = data.get("evaluation_summary")
    changed = old_summary != new_summary
    data["evaluation_summary"] = new_summary
    return changed


def main(argv: list[str]):
    # 新的入参解析: 多路径批处理
    if len(argv) > 1:
        json_paths = [Path(p) for p in argv[1:]]
    else:
        json_paths = JSON_PATHS

    if not json_paths:
        print("未提供需要处理的 JSON 路径。")
        return 1

    any_error = False
    any_changed = False

    for json_path in json_paths:
        if not json_path.exists():
            print(f"[跳过] 文件不存在: {json_path}")
            any_error = True
            continue

        try:
            data = load_json(json_path)
        except Exception as e:
            print(f"[跳过] 读取 JSON 失败: {json_path} -> {e}")
            any_error = True
            continue

        try:
            changed = update_evaluation_summary(data)
        except Exception as e:
            print(f"[跳过] 更新 evaluation_summary 失败: {json_path} -> {e}")
            any_error = True
            continue

        if not changed:
            print(f"[无变化] {json_path}: evaluation_summary 无变化, 不写回。")
            continue

        # 写回: 先备份
        try:
            backup_path = backup_file(json_path)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[失败] 写回失败: {json_path} -> {e}")
            any_error = True
            continue

        any_changed = True
        print(f"[已更新] {json_path} -> 备份: {backup_path}")
        print("新的 per_dimension:")
        for dim, st in data["evaluation_summary"]["per_dimension"].items():
            print(f"  {dim}: count={st['count']} total={st['total_score']} avg={st['average_score']}")

    if any_error and not any_changed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
