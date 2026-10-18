# 训练足球场关键点检测模型

## 结论(先给)

**比篮球容易得多,因为有一个公开数据集 + 预训练模型可以复用。** 模型是
Ultralytics YOLOv8-pose,代码和训练流程开源,本地有 GPU 就能训。

- **路径 A(推荐)**:直接下载 Roboflow 公开数据集 `football-field-detection-f07vi`
  (32 关键点),在此基础上微调。约 1–2 天。
- **路径 B**:在公开数据基础上,补充自己摄像头机位的图,继续微调适配。约 3–5 天。

---

## 1. 关键点方案(已固定)

模型检测 **32 个球场关键点**,顺序已经由 `football-field-detection-f07vi` 数据集
固定,并且和 `pitch_view/configs/soccer.py` 里的 `SoccerPitchConfiguration.vertices`
**一一对应**。这些点覆盖:四个角点、两个禁区(penalty box)、两个小禁区(goal box)、
两个罚球点、中圈、中线和边线交点等。

> 顺序是唯一容易翻车的地方。**复用公开数据集时顺序自动一致,照抄 `vertices` 即可;**
> 只有自采数据标注时才需要按同一顺序手工标注(见第 3 节)。

---

## 2. 路径 A:复用公开数据集微调

```bash
pip install ultralytics roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="你的 ROBOFLOW_API_KEY")
proj = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
ds = proj.version(14).download("yolov8")
print(ds.location)   # 下载后的目录,里面含 train/valid/test 和 data.yaml
```

或直接用项目自带的脚本(自动下载 + 训练):

```bash
export ROBOFLOW_API_KEY=...
python3 scripts/train.py --epochs 100 --imgsz 640 --batch 16 --model yolov8m-pose.pt
```

---

## 3. 路径 B:补充自己机位的数据微调

如果现成模型在你的摄像头视角/场地检出率不够,再补自采数据:

- **采集**:从你自己机位的视频抽帧(覆盖不同时刻、光照、距离),几百张即可。
- **标注**:Roboflow Annotate / CVAT / LabelMe,选「关键点」标注,标注与公开数据集
  **完全相同的 32 个点、相同顺序**,带可见性标记(2 可见 / 1 遮挡 / 0 不在画面)。
- **合并**:把你标注的数据并入公开数据集,一起训练/微调。

关键点标注比目标检测框更费时,每张 32 个点、熟练后约 2–3 分钟/张;但只需要补机位
差异,通常几百张足够。

---

## 4. 训练

`data.yaml`(关键点形状 + 类名,以下载到的 data.yaml 为准):

```yaml
train: /path/to/train/images
val: /path/to/valid/images
kpt_shape: [32, 3]     # 32 个关键点,每个 (x, y, 可见性)
names:
  0: pitch             # 类名以下载到的 data.yaml 为准
```

```bash
yolo pose train \
  data=data.yaml \
  model=yolov8m-pose.pt \
  epochs=100 imgsz=640 batch=16
```

- **模型大小**:`yolov8n/s/m/l/x-pose` 从小到大。Roboflow 官方用 `yolov8x-pose`
  (最大、A100 训练 500 epoch)。你自己微调用 `yolov8m-pose` 或 `yolov8s-pose`
  + 几十个 epoch,单张消费级 GPU 就够、快很多。
- **硬件**:需要 NVIDIA GPU(约 8GB+ 显存跑 m 系列更顺;CPU 会非常慢)。
- **评估**:看 mAP(关键点),更实用的是**单应性重投影误差**——用标注的关键点算
  透视变换,反投影到原图量平均像素误差。

---

## 5. 注意事项

- **License**:Ultralytics YOLO 是 **AGPL-3.0**(传染性)。研究/内部用没问题;若要
  闭源商用,注意衍生作品开源义务,或改用 Apache 2.0 的 RF-DETR。
- **顺序一致**:`kpt_shape` 的 32 个点顺序必须和 `SoccerPitchConfiguration.vertices`
  一致(复用公开数据时天然一致)。

---

## 6. 训练好之后怎么接回本项目

`scripts/train.py` 训完会把 `best.pt` **发布到** `weights/pitch_keypoints.pt`。
当前界面选 YOLO 时读的是 `runs/pose/train/weights/best.pt`（`config/court_detector.yaml` 的 `model_path`）。

其余代码(单应性、top-down 映射、发布)不用改——接口已预留好。

或者,若你把它部署回 Roboflow,也能继续走 `detector_type: inference` + 你的
`model_id`。

---

## 7. 球员检测(自己下载数据训练)

**不要直接用 sports-main 自带的现成 `.pt`**,可以像关键点一样:拉公开数据 → 本地训
YOLO **目标检测**(不是 pose)。

| 项 | 值 |
|---|---|
| 数据集 | [football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) |
| 类别 | `ball` / `goalkeeper` / `player` / `referee`(与 sports-main 一致) |
| 任务 | YOLOv8 **detect**(`--model yolov8m.pt`,不要 `*-pose.pt`) |
| 脚本 | `scripts/train_players.py` |

```bash
export ROBOFLOW_API_KEY=...
pip install ultralytics roboflow

# 下载数据集 v12 + 训练(可改 --version / --epochs / --model)
python3 scripts/train_players.py --download --epochs 100 --model yolov8m.pt

# 已下载过
python3 scripts/train_players.py \
  --data-dir football-players-detection-12 --epochs 100 --model yolov8m.pt
```

训完自动发布到 `weights/players.pt`（GUI「球员推理」默认读这个）。
Ultralytics 中间日志仍在 `runs/player/`。

建议:

- 先公开数据训一版跑通,再混入你自己机位抽帧(同一 4 类框标注)。
- 机位差异大时补几百张通常比硬用官方权重更稳。
- 号码 OCR **不在**这份数据里;EasyOCR 权重也放在同一 `weights/` 目录。

训好后:用 `weights/players.pt` 做球员框检测,取框底边中点,经现有球场
`ViewTransformer` 投到俯视图。
