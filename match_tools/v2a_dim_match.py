import json
import os
from pathlib import Path

# 原始文件路径改为列表，填入多个待处理 json
JSON_PATHS = [
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_vista__video-to-4D-physical_results_v5.json"),
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_traj__video-to-4D-physical_results_v5.json"),
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_recam__video-to-4D-physical_results_v5.json"),
    Path("/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__output_uniform_ex4d__video-to-4D-physical_results_v5.json"),
    # 可继续追加:
    # Path("dimension_description_json/xxx.json"),
]

def extract_dimension(video_path: str) -> str | None:
    """
    从路径中定位 'physical' 段并返回其后的一个段作为 dimension.
    例: /.../physical/dynamics/high_motion/... -> dynamics
    """
    parts = video_path.split('/')
    try:
        idx = parts.index("physical")
        return parts[idx + 1] if idx + 1 < len(parts) else None
    except ValueError:
        return None

def process_file(json_path: Path) -> None:
    if not json_path.exists():
        print(f"未找到文件: {json_path}")
        return

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    video_details = data.get("video_details", [])
    modified = 0

    for item in video_details:
        vp = item.get("video_path")
        if not vp or not isinstance(vp, str):
            continue
        dim = extract_dimension(vp)
        if dim and item.get("dimension") != dim:
            item["dimension"] = dim
            modified += 1

    if modified == 0:
        print(f"{json_path}: 没有需要更新的记录。")
        return

    # 写回
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{json_path}: 已更新 {modified} 条记录")

def main():
    # 批量处理
    for p in JSON_PATHS:
        process_file(p)

if __name__ == "__main__":
    main()