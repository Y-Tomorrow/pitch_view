# pitch_view

把足球场从摄像机画面映射成 **俯视二维球场图**，并可叠加 **球员 / 门将 / 裁判 / 球** 检测、**球衣号码 OCR** 与点击特写。

先检出球场关键点，再用单应（homography）把画面变换到标准球场平面，供后续跟踪、热区、控球分析等使用。架构对齐 [Roboflow sports](https://github.com/roboflow/sports)：球场配置 `SoccerPitchConfiguration` + 变换器 `ViewTransformer` + 可插拔检测器。

也可当普通 Python 工具用（Qt 窗口 / CLI），不必起 ROS2。

---

## 能做什么

| 场景 | 做法 |
|------|------|
| 固定机位 | 标定 ≥4 点保存 JSON，之后每帧用同一套单应 |
| 转播 / 多机位 | YOLO26m-pose 或 Roboflow 关键点模型，逐帧估 H |
| 球员落位 | 「球员推理」：检测框底中点（球用中心）投到俯视图 |
| 球衣号码 | 「号码识别」：胸前裁剪后用 EasyOCR 或 PaddleOCR 认 1～2 位数字 |
| 点击特写 | 点原图检测框 → 放大裁剪，并重写类别 / 置信度 / 号码 |
| 自采关键点 | 从转播抽帧，用 `annotate_keypoints.py` 标 32 点，再和公开集一起微调 |
| 无 ROS | `python3 scripts/run_pitch_view.py`，左原图、右俯视图 |
| 有 ROS2 | `court_detector` 订阅图像，发布标注图、俯视图、关键点 |
| 打包部署 | 带走整个 `weights/` + `config/court_detector.yaml` |

---

## 快速开始

```bash
# 依赖（按需安装）
pip install numpy opencv-python PyQt5 pyyaml
pip install ultralytics          # 本地 YOLO 关键点 / 球员
pip install easyocr              # 球衣号码：EasyOCR（可选）
pip install paddleocr paddlepaddle  # 球衣号码：PaddleOCR 数字（可选）

# Qt 主界面（默认固定机位流程；参数读 config/court_detector.yaml）
python3 scripts/run_pitch_view.py

# 固定机位 CLI
python3 scripts/run_pitch_view.py --cli \
  --calibration data/1_calibration.json data/1.png

# 本地关键点模型
python3 scripts/run_pitch_view.py --cli --detector yolo \
  --model weights/pitch_keypoints.pt data/1.png

# 从转播抽能看到球场的帧，再打开 32 点标定窗口
python3 scripts/extract_pitch_frames.py data/master.mp4
python3 scripts/annotate_keypoints.py
```

打开画面后：

1. 配置检测方法 / 标定或权重 → 打开文件 / 相机 / RTSP  
2. 视频画面下方的进度条可以拖动，松手后从该时间继续播  
3. **球员推理** → 加载 `weights/players.pt`  
4. **号码识别** → 对已检出球员做 OCR（首次需本地 EasyOCR 权重）  
5. **点击检测框** → 右上角特写 + 推理信息面板  

更多命令见 [docs/usage.md](docs/usage.md)。

---

## 权重目录 `weights/`（打包用）

训练与推理共用，**部署时整夹带走**：

| 文件 | 用途 | 写入方 |
|------|------|--------|
| `pitch_keypoints.pt` | 球场 32 关键点，Ultralytics **YOLO26m-pose** | `scripts/train.py` 从 `yolo26m-pose.pt` 微调后发布 |
| `players.pt` | 球 / 门将 / 球员 / 裁判 | `scripts/train_players.py` 训完自动发布 |
| `craft_mlt_25k.pth` | EasyOCR 文字检测 | 首次号码识别下载，或手动放入 |
| `english_g2.pth` | EasyOCR 英文识别 | 同上 |
| `paddle_ocr/` | PaddleOCR 检测 `PP-OCRv5_mobile_det` + 英文识别 `en_PP-OCRv4_mobile_rec` | 界面选「PaddleOCR 数字」后首次下载 |

说明：

- Ultralytics 训练日志仍在 `runs/pose/`、`runs/player/`  
- 训完会把对应 `best.pt` **复制覆盖**到上表文件名  
- 路径常量：`pitch_view/common/weights.py`  
- 详见 [weights/README.md](weights/README.md)

---

## 统一配置 `config/court_detector.yaml`

ROS2 节点与 `run_pitch_view.py` **共用同一份 YAML**。每项参数都有中文备注，主要包括：

| 分组 | 示例键 | 作用 |
|------|--------|------|
| 通用 | `sport` / `scale` / `device` | 项目、俯视图分辨率、CPU/GPU |
| 关键点 | `detector_type` / `model_path` / `min_confidence` | 检测方法与权重 |
| 经典 CV | `canny_*` / `hough_*` | 仅 `classical` |
| 球员 | `player_conf` / `player_iou` / `player_nms_*` | 检测阈值与去重 |
| 号码 | `jersey_ocr_engine` / `jersey_ocr_min_conf` / `jersey_torso_*` | EasyOCR 或 Paddle、置信度、胸前裁剪 |
| 特写 | `closeup_pad_ratio` / `closeup_track_iou` | 点击放大与跟踪 |

改完后重启 GUI 或 ROS 节点生效。也可：

```bash
python3 scripts/run_pitch_view.py --config /path/to/custom.yaml
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
  │     过滤 (0,0) / 画外 / 贴边 / 低可见度
  │     少于 4 个不共线对应点 → 求不出 H，不贴俯视图、不投球员
  │
  ├─ ViewTransformer
  │     H = findHomography(keypoints, target × scale)   # 8 自由度，至少 4 对点
  │     warped = warpPerspective(frame, H, 球场像素尺寸)
  │
  ├─ （可选）球员 YOLOv8 detect → ball / GK / player / ref
  │     跨类 NMS / 后处理去重（减少 player+GK 叠框）
  │     脚点（框底中点）或球心 → 同一张 H 投到俯视图
  │
  ├─ （可选）球衣号码
  │     胸前裁剪 → 放大、CLAHE、锐化，并试反色
  │     EasyOCR（数字白名单）或 PaddleOCR（检测 + 整图识别兜底）
  │     只留 1～2 位数字里置信度最高的结果
  │
  └─ compose_topdown + Qt 特写面板
```

核心假设：**球场是平面**。三个不共线的点就能定下这个平面，但这里要算的是像素和球场坐标之间的 **单应矩阵**（3×3，去掉尺度后 8 个未知数）。每一对点只给两个方程，所以至少要 **4 对点**。3 对点只能定仿射变换，透视相机下定不住这张映射。点数多于 4 时 `findHomography` 做最小二乘折中。

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
| `yolo` | 本地 YOLO26m-pose `.pt` | 默认 `weights/pitch_keypoints.pt` |
| `inference` | Roboflow 本机缓存权重 | `inference` + API key |
| `roboflow` | 云端 HTTP | `inference-sdk` + API key |
| `classical` | Canny + Hough 四角 | OpenCV |

Qt 里 YOLO / 球员 / OCR 均在 **独立 spawn 子进程**，避免 torch 与 Qt 抢运行时（WSLg 上常见 segfault）。

---

## 试过的算法和模型

下面是这个仓库里实际接通过的方案。关键点的两行 mAP 来自两套不同的验证集，不能横着比大小。

### 球场关键点

| 方案 | 优点 | 缺点 | 在这套转播上的表现 |
|------|------|------|--------------------|
| 固定单应 `homography` | 标定一次即可，不跑模型 | 相机、变焦一变，整张 H 就废 | 只适合固定机位。本场转播机位一直在变，不能靠一张标定走完全场 |
| Canny + Hough `classical` | 不需要权重 | 只找四个角，抗不了人群、台标、草皮阴影 | 给篮球那种干净边线。足球转播上既不稳，也给不出 32 个点 |
| Roboflow 公开模型 `inference` / `roboflow` | 不用自己标；本机缓存比云端快 | 云端要 API 和网络。权重是 DFL 转播素材上训的 | 和下面的公开微调是同一类分布偏移：换到这场 Setanta 画面，点经常对不上 |
| YOLOv8m-pose，公开集微调 200 epoch → `weights/pitch_keypoints.pt` | 公开验证集上很高：关键点 mAP50 **0.995**，mAP50-95 **0.839**（34 张公开验证图） | 没见过这场的机位、灯光和包装 | 用它看 `data/master.mp4`：开场近景、特写经常是 0 个关键点，单应算不出来；宽角战术镜头才能出点 |
| YOLOv8m-pose，在上一版上再微调 50 epoch → `runs/pose/train/weights/best.pt` | 训练集混入本场 768 张标注。验证集关键点 precision **0.955**、recall **0.920**、mAP50 **0.941**、mAP50-95 **0.744** | 验证集比公开集那 34 张难，所以 mAP 数字低于上一行 | 同一套混合数据上的 v8 对照 |
| YOLO26m-pose，公开集 + 本场 768 张，100 epoch → `weights/pitch_keypoints.pt` | 同一验证集：precision **0.994**、recall **0.945**、mAP50 **0.955**、mAP50-95 **0.904** | 比 v8 慢一档；播放速度跟推理走，左右画面同一帧 | 当前关键点权重。日志在 `runs/pose/yolo26m/results.csv` |

v8 对照起点是 `yolov8m-pose.pt`，YOLO26 起点是 `yolo26m-pose.pt`。都是 32 点、输入 640、RTX 4060、batch 8。预训练底模下到 `weights/`（例如 `weights/yolo26m-pose.pt`），训完的 `best.pt` 再复制为 `weights/pitch_keypoints.pt`。界面 `device: cuda` 时走 GPU。

播放时左图和俯视图一起更新：当前帧推理结束后才解码下一帧，所以两边同步，帧率由模型决定。

### 球员检测

用的是 **YOLOv8m detect**（`yolov8m.pt` 训 100 epoch）→ `weights/players.pt`。类别 `ball` / `GK` / `player` / `ref`。公开验证集 precision **0.935**、recall **0.832**、mAP50 **0.861**、mAP50-95 **0.592**。

框能检出来，落点仍靠球场单应：人用框底中点，球用框中心。同一人会同时出 `player` 和 `GK`，所以做了跨类 NMS。当时用旧关键点权重看本场，有的画面能检出大约 9～11 个球员框，但关键点仍是 0，俯视图贴不上去。

### 球衣号码

两条都是「胸前裁剪 → 放大 → 只留 1～2 位数字」，没有单独标过号码准确率。

| 方案 | 优点 | 缺点 | 画面上的感觉 |
|------|------|------|----------------|
| EasyOCR `english_g2` | 安装路径短，有数字白名单 | 通用英文模型，小字、彩色球衣、低对比容易碎 | 明显的号码也经常认不出 |
| PaddleOCR：检测 `PP-OCRv5_mobile_det`，识别 `en_PP-OCRv4_mobile_rec` | 同一批画面上比 EasyOCR 稳。先看彩色再试反色；拆开的相邻数字会合并；检测不够有把握时整图再认一次 | 手机版英文模型，不是球衣专用。CPU 版 Paddle 会慢 | 比 EasyOCR 好一截，清楚的号码仍会漏或错。这条路径改完后没有再统计整场准确率 |

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

这是 **固定机位的 4 点单应**，和下面的 32 点数据集标定不是同一件事。机位一变，这张 JSON 就失效。

---

## 32 点数据集标定

公开模型在本段转播上会漏点。做法是从视频里抽出本场机位的画面，按与 `football-field-detection-14` 相同的 32 点顺序标注，再和公开集一起微调。

```bash
python3 scripts/extract_pitch_frames.py data/master.mp4
python3 scripts/annotate_keypoints.py
```

`extract_pitch_frames.py` 顺序读视频，丢掉球员特写、看台和几乎重复的帧，只留能看到标线的画面。默认写到 `data/pitch_dataset/images/`，清单是 `manifest.csv`。

`annotate_keypoints.py` 是 Qt 标定窗口：

| 区域 | 内容 |
|------|------|
| 左 | 按文件名顺序的待标图片 |
| 右 | 二维球场，32 个点标着 **0–31**（与 `SoccerPitchConfiguration.vertices`、公开集一致） |
| 下 | 两行按钮，每行 16 个，对应 0–15 和 16–31 |

| 操作 | 作用 |
|------|------|
| 「打开图片…」 | 选一张图作为起点；同一目录其余图片仍按文件名排在前后。选 `train` 或 `train/images` 都会读同级 `labels/*.txt` |
| 「权重…」 / **W** | 用 `weights/pitch_keypoints.pt`（可换）给当前图预标，再拖动绿点微调。低置信度的点留空 |
| 左键 / 右键 / **N** | 可见 / 遮挡 / 不在画面。标完当前点后下一个编号 +1 |
| 下方按钮或右侧编号 | 跳到该点。没点过的中间点保持空白，不会被写成「不在画面」 |
| **Z** / **A** **D** / 空格 | 撤销、上一张下一张、顺序播放 |

换图或关闭窗口时，改过的那张会直接写入图片目录同级的 `labels/`：

- `<同名>.txt`：与 `football-field-detection-14` 相同的 YOLO 关键点一行  
  `类别 cx cy w h` + 32 组 `x y 可见性`，坐标 0–1，类别 `pitch`（0）。  
  可见性 2 / 1 / 0。没标的点和按 **N** 的点在这份 `.txt` 里都是 `0 0 0`。
- `<同名>.json`：只给窗口记住哪些点还没标，训练不用。一张图一个点都没标则不生成文件。只翻看、不改点，不会重写已有 `.txt`。

标完后把图片和 `.txt` 并入 `football-field-detection-14` 的 `train/`（公开集 `valid/` 留作验证），用 YOLO26 预训练权重微调。`weights/pitch_keypoints.pt` 若是另一代 YOLO，不能当作 YOLO26 的起点。

```bash
cp -n data/pitch_dataset/images/* football-field-detection-14/train/images/
cp -n data/pitch_dataset/labels/*.txt football-field-detection-14/train/labels/

python3 scripts/train.py \
  --data-dir /home/yy/pitch_view/football-field-detection-14 \
  --model /home/yy/pitch_view/weights/yolo26m-pose.pt \
  --epochs 100 --imgsz 640 --batch 8 --name yolo26m
```

中断后续训：

```bash
yolo pose train resume model=runs/pose/yolo26m/weights/last.pt
```

点的物理含义见 [docs/annotation.md](docs/annotation.md)。操作细节见 [docs/usage.md](docs/usage.md)。

---

## 目录结构

```
pitch_view/
├── pitch_view/
│   ├── common/
│   │   ├── detector.py      # 关键点检测器工厂
│   │   ├── view.py          # ViewTransformer
│   │   ├── weights.py       # weights/ 路径与训后发布
│   │   └── settings.py      # 加载 court_detector.yaml
│   ├── configs/soccer.py
│   ├── annotators/soccer.py
│   └── court_detector_node.py
├── scripts/
│   ├── run_pitch_view.py    # Qt / CLI 主入口
│   ├── calibrate_gui.py / calibrate.py
│   ├── train.py             # → weights/pitch_keypoints.pt
│   ├── train_players.py     # → weights/players.pt
│   ├── extract_pitch_frames.py   # 转播抽帧 → data/pitch_dataset/
│   ├── annotate_keypoints.py     # 32 点标定窗口
│   ├── dump_keypoints.py / detect_image.py
├── weights/                 # 打包权重目录
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
pip install easyocr              # 号码 OCR：EasyOCR
pip install paddleocr paddlepaddle  # 号码 OCR：PaddleOCR 数字
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

**球场关键点**用 Ultralytics **YOLO26m-pose**（32 点，`kpt_shape: [32, 3]`）→ `weights/pitch_keypoints.pt`。`--model` 写成 `weights/` 下的路径，避免预训练文件下到仓库根目录：

```bash
export ROBOFLOW_API_KEY=...
python3 scripts/train.py --download
python3 scripts/train.py --data-dir football-field-detection-14 \
  --epochs 100 --imgsz 640 --batch 8 \
  --model weights/yolo26m-pose.pt --name yolo26m
```

**球员检测**（detect，勿用 `*-pose.pt`）→ `weights/players.pt`：

```bash
python3 scripts/train_players.py --download --epochs 100 --model yolov8m.pt
```

中间产物在 `runs/`；正式推理权重以 `weights/` 为准。细节见 [docs/training.md](docs/training.md)。

---

## 单应说明（简要）

代码：`pitch_view/common/view.py`。

- 平面由 3 个不共线的点确定；单应矩阵有 8 个自由度，至少要 4 对像素↔球场坐标  
- `findHomography` 拟合；点数 > 4 时为最小二乘折中  
- `warpPerspective` 整图变换；贴边约 1.2% 不参与取样，减轻外围伪影  
- 少于 4 个点时不生成俯视图，球员也不投影  
- 像素级难完全重合：镜头畸变、球场比例与 120 m×70 m 模板的差异、漆线宽度等  

用到的主要库：OpenCV（几何与图像）、NumPy、PyQt5（界面）、Ultralytics YOLO（关键点为 **YOLO26m-pose**，球员为 YOLOv8m detect）、EasyOCR / PaddleOCR + PaddlePaddle（号码）、PyYAML（配置）。可选：Roboflow `inference` / `inference-sdk`、`roboflow`（下载数据集）、ROS 2 + `cv_bridge`。YOLO、Paddle、EasyOCR 在 Qt 里跑在 spawn 子进程中。有 GPU 时 `config/court_detector.yaml` 的 `device` 设为 `cuda`。

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
| [docs/usage.md](docs/usage.md) | 推理、固定机位标定、32 点标注与训练命令 |
| [docs/training.md](docs/training.md) | 数据集与训练细节 |
| [docs/annotation.md](docs/annotation.md) | 关键点语义与标注约定 |
| [weights/README.md](weights/README.md) | 权重文件清单 |
| [config/court_detector.yaml](config/court_detector.yaml) | 全部可调参数（含备注） |

---

## 引用

| 来源 | 用途 |
|------|------|
| [FullMatchShows](https://fullmatchshows.com/) | 整场比赛回放，用作抽帧和关键点数据集 |
| [PFA_radar-2025](https://github.com/Y-Tomorrow/PFA_radar-2025) | 单目画面标定后映射到二维平面的思路（相机视图与地图视图成对点，求出变换） |
| [Roboflow sports](https://github.com/roboflow/sports) | 球场关键点配置、单应变换与检测器结构 |

```bibtex
@misc{fullmatchshows,
  title        = {FullMatchShows},
  howpublished = {\url{https://fullmatchshows.com/}},
  note         = {Full match replay videos used as the frame dataset}
}

@misc{pfa_radar_2025,
  title        = {PFA\_radar-2025},
  author       = {{Y-Tomorrow}},
  howpublished = {\url{https://github.com/Y-Tomorrow/PFA_radar-2025}},
  note         = {Monocular camera to 2D map transform}
}

@misc{roboflow_sports,
  title        = {sports: computer vision and sports},
  author       = {{Roboflow}},
  howpublished = {\url{https://github.com/roboflow/sports}}
}
```

## 许可

MIT（见 `package.xml`）。公开球场 / 球员数据遵循 Roboflow 数据集许可（CC BY 4.0）。Ultralytics YOLO 为 AGPL-3.0，闭源商用需自行评估。
