# pitch_view

把足球场从摄像机画面映射成 **俯视二维球场图**，并可叠加 **球员 / 门将 / 裁判 / 球** 检测、**球衣号码 OCR** 与点击特写。

先检出球场关键点，再用单应（homography）把画面变换到标准球场平面，供后续跟踪、热区、控球分析等使用。架构对齐 [Roboflow sports](https://github.com/roboflow/sports)：球场配置 `SoccerPitchConfiguration` + 变换器 `ViewTransformer` + 可插拔检测器。

也可当普通 Python 工具用（Qt 窗口 / CLI），不必起 ROS2。

---

## 能做什么

| 场景 | 做法 |
|------|------|
| 固定机位 | 标定 ≥4 点保存 JSON，之后每帧用同一套单应 |
| 转播 / 多机位 | YOLO-pose 或 Roboflow 关键点模型，逐帧估 H |
| 球员落位 | 「球员推理」：检测框底中点（球用中心）投到俯视图 |
| 球衣号码 | 「号码识别」：对检出球员裁胸前区域做 EasyOCR |
| 点击特写 | 点原图检测框 → 放大裁剪，并重写类别 / 置信度 / 号码 |
| 无 ROS | `python3 scripts/run_homography.py`，左原图、右俯视图 |
| 有 ROS2 | `court_detector` 订阅图像，发布标注图、俯视图、关键点 |
| 打包部署 | 带走整个 `models/` + `config/court_detector.yaml` |

---

## 快速开始

```bash
# 依赖（按需安装）
pip install numpy opencv-python PyQt5 pyyaml
pip install ultralytics          # 本地 YOLO 关键点 / 球员
pip install easyocr              # 球衣号码（可选）

# Qt 主界面（默认固定机位流程；参数读 config/court_detector.yaml）
python3 scripts/run_homography.py

# 固定机位 CLI
python3 scripts/run_homography.py --cli \
  --calibration data/1_calibration.json data/1.png

# 本地关键点模型
python3 scripts/run_homography.py --cli --detector yolo \
  --model models/pitch_keypoints.pt data/1.png
```

打开画面后：

1. 配置检测方法 / 标定或权重 → 打开文件 / 相机 / RTSP  
2. **球员推理** → 加载 `models/players.pt`  
3. **号码识别** → 对已检出球员做 OCR（首次需本地 EasyOCR 权重）  
4. **点击检测框** → 右上角特写 + 推理信息面板  

更多命令见 [docs/usage.md](docs/usage.md)。

---

## 权重目录 `models/`（打包用）

训练与推理共用，**部署时整夹带走**：

| 文件 | 用途 | 写入方 |
|------|------|--------|
| `pitch_keypoints.pt` | 球场 32 关键点 YOLO-pose | `scripts/train.py` 训完自动发布 |
| `players.pt` | 球 / 门将 / 球员 / 裁判 | `scripts/train_players.py` 训完自动发布 |
| `craft_mlt_25k.pth` | EasyOCR 文字检测 | 首次号码识别下载，或手动放入 |
| `english_g2.pth` | EasyOCR 英文识别 | 同上 |

说明：

- Ultralytics 训练日志仍在 `runs/pose/`、`runs/player/`  
- 训完会把对应 `best.pt` **复制覆盖**到上表文件名  
- 路径常量：`pitch_view/common/weights.py`  
- 详见 [models/README.md](models/README.md)

---

## 统一配置 `config/court_detector.yaml`

ROS2 节点与 `run_homography.py` **共用同一份 YAML**。每项参数都有中文备注，主要包括：

| 分组 | 示例键 | 作用 |
|------|--------|------|
| 通用 | `sport` / `scale` / `device` | 项目、俯视图分辨率、CPU/GPU |
| 关键点 | `detector_type` / `model_path` / `min_confidence` | 检测方法与权重 |
| 经典 CV | `canny_*` / `hough_*` | 仅 `classical` |
| 球员 | `player_conf` / `player_iou` / `player_nms_*` | 检测阈值与去重 |
| 号码 | `jersey_ocr_min_conf` / `jersey_torso_*` | OCR 与胸前裁剪 |
| 特写 | `closeup_pad_ratio` / `closeup_track_iou` | 点击放大与跟踪 |

改完后重启 GUI 或 ROS 节点生效。也可：

```bash
python3 scripts/run_homography.py --config /path/to/custom.yaml
```

加载逻辑：`pitch_view/common/settings.py`。

---

## 技术链路

```
相机帧
  │
  ├─ 球场关键点 detect(frame) → (keypoints, target)
  │     keypoints: 画面像素 (N, 2)
  │     target:    模板球场坐标 cm (N, 2)，与 keypoints 同序
  │     过滤 (0,0) / 画外 / 贴边 / 低可见度；N < 4 无法求 H
  │
  ├─ ViewTransformer
  │     H = findHomography(keypoints, target × scale)
  │     warped = warpPerspective(frame, H, 球场像素尺寸)
  │
  ├─ （可选）球员 YOLO detect → 框 + 类别
  │     跨类 NMS / 后处理去重（减少 player+GK 叠框）
  │     脚点（框底中点）或球心 → 投到俯视图
  │
  ├─ （可选）EasyOCR → 球衣 1～2 位数字
  │
  └─ compose_topdown + Qt 特写面板
```

核心假设：**球场是平面**，针孔相机下平面↔平面可用一张 3×3 单应表示。

---

## 球场模板（32 关键点）

写在 `pitch_view/configs/soccer.py` 的 `SoccerPitchConfiguration`：

- 尺寸：**12000 × 7000 cm**（120 m × 70 m）
- 与公开数据集 `football-field-detection-f07vi` 一致  
- `vertices` 顺序必须与模型输出一致，否则单应错乱  
- 语义表见 [docs/annotation.md](docs/annotation.md)

`scale`：俯视图每厘米多少像素，默认读 YAML（通常 `0.05`）。

篮球：`BasketballCourtConfiguration` + `annotators/basketball.py`，同一套 `ViewTransformer`。

---

## 检测器

工厂：`create_detector()`（`pitch_view/common/detector.py`）。

```text
detect(frame) -> (keypoints, target)
```

| 类型 | 原理 | 依赖 |
|------|------|------|
| `homography` | 读标定 JSON，每帧固定点对 | 标定文件 |
| `yolo` | 本地 YOLO-pose `.pt` | `ultralytics` + `models/pitch_keypoints.pt` |
| `inference` | Roboflow 本机缓存权重 | `inference` + API key |
| `roboflow` | 云端 HTTP | `inference-sdk` + API key |
| `classical` | Canny + Hough 四角 | OpenCV |

Qt 里 YOLO / 球员 / OCR 均在 **独立 spawn 子进程**，避免 torch 与 Qt 抢运行时（WSLg 上常见 segfault）。

---

## 球员 · 号码 · 特写

- **球员模型类别**：`ball` / `GK` / `player` / `ref`  
- **置信度等**：`player_conf`（默认 0.25）、`player_iou`、跨类 NMS 等见 YAML  
- **号码**：未开启 / 未能识别 / `#号` 三种文案分开显示，避免和「球员未检出」混淆  
- **特写**：点检测框放大；视频流中按 IoU 跟踪同一框  

---

## 标定（固定机位）

机位、变焦、分辨率不变时，不必每帧跑关键点模型。

```bash
python3 scripts/calibrate_gui.py --source 0          # 相机
python3 scripts/calibrate_gui.py --source data/1.png # 单图
python3 scripts/calibrate.py data/1.png \
  --points "x1,y1;x2,y2;x3,y3;x4,y4"                 # 无头
```

JSON 含 `source`（像素）与 `target`（cm）。在 YAML 里设：

```yaml
detector_type: homography
homography_file: "data/1_calibration.json"
```

---

## 目录结构

```
pitch_view/
├── pitch_view/
│   ├── common/
│   │   ├── detector.py      # 关键点检测器工厂
│   │   ├── view.py          # ViewTransformer
│   │   ├── weights.py       # models/ 路径与训后发布
│   │   └── settings.py      # 加载 court_detector.yaml
│   ├── configs/soccer.py
│   ├── annotators/soccer.py
│   └── court_detector_node.py
├── scripts/
│   ├── run_homography.py    # Qt / CLI 主入口
│   ├── calibrate_gui.py / calibrate.py
│   ├── train.py             # → models/pitch_keypoints.pt
│   ├── train_players.py     # → models/players.pt
│   ├── dump_keypoints.py / detect_image.py
├── models/                  # 打包权重目录
├── config/court_detector.yaml
├── launch/court_detector.launch.py
├── docs/usage.md · annotation.md · training.md
└── test/
```

---

## 安装

```bash
pip install numpy opencv-python PyQt5 pyyaml
pip install ultralytics          # YOLO 关键点 / 球员
pip install easyocr              # 号码 OCR
pip install inference            # Roboflow 本地推理（可选，Py3.8–3.11）
pip install roboflow             # 下载公开数据集（训练时）
```

Roboflow：

```bash
export ROBOFLOW_API_KEY=...
export MODEL_CACHE_DIR="$HOME/.roboflow/cache"
mkdir -p "$HOME/.roboflow/cache"
```

ROS2：

```bash
sudo apt install ros-$ROS_DISTRO-cv-bridge
cd ~/ws && colcon build --packages-select pitch_view && source install/setup.bash
```

---

## ROS2

```bash
ros2 launch pitch_view court_detector.launch.py
# 或
ros2 run pitch_view court_detector --ros-args -r image:=/camera/image_raw
```

| Topic | 类型 | 含义 |
|-------|------|------|
| `image`（订） | `sensor_msgs/Image` | 输入帧 |
| `~/annotated` | `sensor_msgs/Image` | 画了关键点的原图 |
| `~/pitch_map` | `sensor_msgs/Image` | 俯视 warp |
| `~/keypoints` | `geometry_msgs/Polygon` | 像素关键点 |

参数见 `config/court_detector.yaml`（launch 加载）。

---

## 训练

**球场关键点**（pose）→ `models/pitch_keypoints.pt`：

```bash
export ROBOFLOW_API_KEY=...
python3 scripts/train.py --download
python3 scripts/train.py --data-dir football-field-detection-14 \
  --epochs 100 --model yolov8m-pose.pt
```

**球员检测**（detect，勿用 `*-pose.pt`）→ `models/players.pt`：

```bash
python3 scripts/train_players.py --download --epochs 100 --model yolov8m.pt
```

中间产物在 `runs/`；正式推理权重以 `models/` 为准。细节见 [docs/training.md](docs/training.md)。

---

## 单应说明（简要）

代码：`pitch_view/common/view.py`。

- `findHomography` 拟合；点数 > 4 时为最小二乘折中  
- `warpPerspective` 整图变换；贴边约 1.2% 不参与取样，减轻外围伪影  
- 像素级难完全重合：畸变、球场比例与模板差异、漆线宽度等（物理限制）

---

## 测试

```bash
colcon test --packages-select pitch_view
# 或
python3 -m pytest test/
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [docs/usage.md](docs/usage.md) | 标定、推理、训练命令 |
| [docs/training.md](docs/training.md) | 数据集与训练细节 |
| [docs/annotation.md](docs/annotation.md) | 关键点语义与标注约定 |
| [models/README.md](models/README.md) | 权重文件清单 |
| [config/court_detector.yaml](config/court_detector.yaml) | 全部可调参数（含备注） |

---

## 引用

球场关键点配置、单应变换与检测器结构参考 [Roboflow sports](https://github.com/roboflow/sports)（MIT）。

```bibtex
@misc{roboflow_sports,
  title        = {sports: computer vision and sports},
  author       = {{Roboflow}},
  howpublished = {\url{https://github.com/roboflow/sports}},
  note         = {MIT License}
}
```

## 许可

MIT（见 `package.xml`）。公开球场 / 球员数据遵循 Roboflow 数据集许可（CC BY 4.0）。Ultralytics YOLO 为 AGPL-3.0，闭源商用需自行评估。
