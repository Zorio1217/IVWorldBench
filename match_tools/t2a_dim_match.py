import json
import os
import re
import shutil
import difflib
from typing import Any, Dict, List, Optional, Tuple

RESULTS_JSON = "/data2/luowei/phywordbench/dimension_description_json/physics_reasoning__4d-fy__text_to_4D_physical_results_v5.json"
PROMPT_JSON  = "/data2/luowei/Videos/condition_to_4D/text-to-any/prompt/physical/phygenbench_sample.json"

def load_json_tolerant(path: str) -> Any:
    # 允许文件头尾有杂字符（如多余的数字、日志、BOM、Markdown栈等）
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    # 去掉 UTF-8 BOM
    s = s.lstrip("\ufeff")
    # 找到第一个 '{' 或 '[' 开始位置
    m = re.search(r"[\{\[]", s)
    if not m:
        raise ValueError(f"无法在 {path} 中定位 JSON 起始字符")
    start = m.start()
    # 从结尾反向找最后一个配对结束符
    end_brace = s.rfind("}")
    end_brack = s.rfind("]")
    end = max(end_brace, end_brack)
    if end == -1:
        raise ValueError(f"无法在 {path} 中定位 JSON 结束字符")
    core = s[start:end+1]
    return json.loads(core)

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_records_with_caption(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "caption" in obj[0]:
            return obj
    if isinstance(obj, dict):
        # 常见容器键
        for key in ("data", "items", "records", "list"):
            v = obj.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict) and "caption" in v[0]:
                return v
        # 广搜一层
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "caption" in v[0]:
                return v
    raise ValueError("未在提示库 JSON 中找到包含 caption 的记录列表")

def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.rstrip(" .!?:;，。！？：；")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s.casefold()

def build_caption_index(records: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    exact: Dict[str, str] = {}
    norm: Dict[str, str] = {}
    for r in records:
        cap = r.get("caption")
        cat = r.get("main_category")
        if not isinstance(cap, str) or cat is None:
            continue
        exact.setdefault(cap, cat)
        norm.setdefault(normalize_text(cap), cat)
    return exact, norm

def lookup_category(prompt: str, exact: Dict[str, str], norm: Dict[str, str]) -> Optional[str]:
    if prompt in exact:
        return exact[prompt]
    simple = prompt.rstrip(" .!?:;，。！？：；")
    if simple in exact:
        return exact[simple]
    n = normalize_text(prompt)
    if n in norm:
        return norm[n]
    # 高阈值模糊匹配，避免误配
    keys = list(norm.keys())
    cand = difflib.get_close_matches(n, keys, n=1, cutoff=0.985)
    if cand:
        return norm[cand[0]]
    return None

def main():
    if not os.path.isfile(RESULTS_JSON):
        raise FileNotFoundError(f"结果文件不存在: {RESULTS_JSON}")
    if not os.path.isfile(PROMPT_JSON):
        raise FileNotFoundError(f"提示库文件不存在: {PROMPT_JSON}")

    results = load_json_tolerant(RESULTS_JSON)
    prompt_obj = load_json_tolerant(PROMPT_JSON)

    records = extract_records_with_caption(prompt_obj)
    exact, norm = build_caption_index(records)

    vd = results.get("video_details")
    if not isinstance(vd, list):
        raise ValueError("结果 JSON 中未找到 video_details 列表")

    matched = 0
    missing = 0
    for item in vd:
        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            item["dimension"] = None
            missing += 1
            continue
        cat = lookup_category(prompt, exact, norm)
        if cat is None:
            item["dimension"] = None
            missing += 1
        else:
            item["dimension"] = cat
            matched += 1

    print(f"匹配到: {matched}, 未匹配: {missing}, 总计: {len(vd)}")

    # 备份并写回
    # backup = RESULTS_JSON + ".bak"
    # shutil.copy2(RESULTS_JSON, backup)
    save_json(RESULTS_JSON, results)
    print(f"已写回: {RESULTS_JSON}")
    # print(f"已备份: {backup}")

if __name__ == "__main__":
    main()