# 轻量化视频评测套件（九个维度）

本仓库只保留九个指标脚本，并使用 `runner.py` 统一调度。**不在仓库中内置** SLAM / 深度学习第三方工程的源码（避免许可证与版权问题），仅用文档链接说明如何自备轨迹或其它预测结果。

推理设备：**在有 NVIDIA GPU + CUDA PyTorch 的主机上优先跑 `cuda`**。若当前环境没有 CUDA，会自动退回 CPU/MPS（会打印提示）；你本地在无 GPU 的机器上仅能验证流程，无法在真实速度下调试 GPU kernel。

---

## 维度与模块名（`--dimension`）

| 模块名 | 说明 |
| --- | --- |
| `scene_memory` | 对称帧 PSNR + LPIPS，衡量长程外观记忆 |
| `object_amount_memory` | 抽检帧上做目标计数，惩罚计数波动 |
| `object_shape_memory` | ResNet18 全局嵌入首尾相似度 |
| `camera_transform_error` | **仅用 JSON 自带的轨迹**：`translations_*`/`poses_*` |
| `camera_rotation_stability` | ORB/Essential 矩阵估计相邻帧转动幅度方差倒数 |
| `clip_iqa` | PyIQA 的 CLIP-IQA NR 度量族 |
| `clip_aesthetic` | TorchMetrics `CLIPScore`，单条正向美学措辞 |
| `consistency_3d` | 时间间隔抽样帧对上 LPIPS，再映射为 \(\exp(-\text{LPIPS})\) |
| `style_consistency` | VGG 浅层 Gram 统计，惩罚帧间 Gram 波动 |

列出当前存在的模块：

```bash
python runner.py --list_dimensions
```

---

## 环境

```bash
conda create -n bench python=3.10 -y
conda activate bench

# ★ CUDA 主机：请先安装与你的驱动匹配的 Torch（示例 cu121）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd /path/to/benchmark
pip install -r requirements.txt
```

`object_amount_memory` 默认尝试 `YOLO_WEIGHTS=yolov8n.pt`（Ultralytics 会按需下载）。

---

## 运行示例

单视频：

```bash
DEVICE=cuda:0 python runner.py \
  --dimension clip_iqa \
  --model_name my_run \
  --video_path ./sample.mp4
```

文件夹（每个视频一行 JSON）：

```bash
DEVICE=cuda:0 python runner.py \
  --dimension style_consistency \
  --model_name my_run \
  --dataset_path ./clips/ \
  --recursive
```

`camera_transform_error` 需要在数据集 JSON **或 runner 的中间 JSON** 里为条目提供 **`auxiliary_info` 字典**：

```json
"auxiliary_info": {
  "translations_gt": [[0,0,0], ...],
  "translations_pred": [[dx,dy,dz], ...]
}
```

也可以用 `poses_gt` / `poses_pred`（每帧 4×4 矩阵列表），脚本会自动取 \(t=\)矩阵平移向量。  
轨迹预测请在 **外部工程**完成，再把结果写进 JSON。常用参考（任选其一，均需自行遵从其开源协议）：  

-相机/场景几何：  

- [Princeton VL / DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)  
- [COLMAP](https://github.com/colmap/colmap)  
- [dust3r](https://github.com/naver/dust3r)（仅作第三方示例）

`evaluate.sh` 占位示例：

```bash
export DATA_JSON=/abs/dataset.json
export MODEL=my_model_name
export DIMENSION=clip_aesthetic
export DEVICE=cuda:0
bash evaluate.sh --prompt "optional caption override"
```

---

## CUDA 主机 vs 仅有 CPU

- **`--device cuda:0`**：若 Torch 检测到 CUDA，`BENCH_TORCH_DEVICE=cuda:0`，所有 TorchMetrics / PyTorch 模型跑在显卡上。
- **无 CUDA**：自动退回 **`cpu`** 并告警；CLIP / LPIPS / YOLO 仍可跑只是非常慢。
- **Apple Silicon**：可把 `--device mps`。

---

## 输出

- runner 在中间目录生成 `dimension_description_json/<dimension>.json`；
- 各指标还会在同级目录导出 `*-results.json`，并在 `results/*.npz` 写简略汇总。
