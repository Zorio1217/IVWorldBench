import json
import re

# 读取 physical_caption_0909.json
with open('physical_caption_0909.json', 'r') as f:
    captions = json.load(f)

input_dir = 'dimension_description_json/physics_reasoning__CamI2V__image-any-physical-image-to-4D_results_v5.json'
# 读取 physics_reasoning__Diffusion_as_Shader__image-any-physical-image-to-4D_results.json
with open(input_dir, 'r') as f:
    results = json.load(f)

# 构建 uuid 到 dimension 的映射
uuid_to_dimension = {}
for item in captions:
    video_path = item['video']
    # 提取 uuid
    match = re.search(r'/([0-9a-f]{64})_', video_path)
    if match:
        uuid = match.group(1)
        # 提取 dimension
        dim_match = re.search(r'/physical/([^/]+)/', video_path)
        if dim_match:
            dimension = dim_match.group(1)
            uuid_to_dimension[uuid] = dimension

# 遍历 results，匹配 uuid，添加 dimension 字段
for video_detail in results.get('video_details', []):
    video_path = video_detail.get('video_path', '')
    match = re.search(r'/([0-9a-f]{64})_', video_path)
    if match:
        uuid = match.group(1)
        if uuid in uuid_to_dimension:
            video_detail['dimension'] = uuid_to_dimension[uuid]
            print(f"Added dimension '{uuid_to_dimension[uuid]}' to video_detail")

# 保存结果
with open(input_dir.replace('.json', '_dim.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)