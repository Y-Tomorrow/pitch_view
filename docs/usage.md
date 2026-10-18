# 用法

统一配置见仓库根目录 [`config/court_detector.yaml`](../config/court_detector.yaml)
（球员置信度、NMS、OCR、特写等均可改；GUI / CLI / ROS 共用）。

```bash
python3 scripts/run_pitch_view.py --config config/court_detector.yaml
```

## 标定(固定机位)

```bash
# Qt GUI:左图直播/右图二维图,「开始标定」暂停一帧,左右各点 4 点后「计算保存」
python3 scripts/calibrate_gui.py --source 0               # 相机 0
python3 scripts/calibrate_gui.py --source /path/match.mp4 # 视频文件
python3 scripts/calibrate_gui.py --source rtsp://...      # 网络流
python3 scripts/calibrate_gui.py --source data/1.png      # 单张图

# 无头兜底:直接给四角像素点(左上→右上→右下→左下)
python3 scripts/calibrate.py data/1.png --points "x1,y1;x2,y2;x3,y3;x4,y4"
```

标定完成后在 `config/court_detector.yaml` 设 `detector_type: homography` 和
`homography_file: "xxx_calibration.json"`。

## 固定机位推理(调 JSON 得到俯视图)

```bash
# 单张图 -> 生成 *_warped.jpg(俯视图)和 *_annotated.jpg(标定点)
python3 scripts/run_pitch_view.py --calibration data/1_calibration.json data/1.png

# 视频 -> 生成 *_warped.mp4
python3 scripts/run_pitch_view.py --calibration calib.json --video match.mp4 \
    --scale 0.05 --out-dir out/

# Qt 窗口:可选输入源(文件/相机/RTSP),左原图右俯视图
# 打开视频后，画面下方进度条可拖动跳转
python3 scripts/run_pitch_view.py --calibration calib.json
```

推理权重统一放在 `weights/`（打包带走即可）:

| 文件 | 用途 |
|------|------|
| `weights/pitch_keypoints.pt` | 球场关键点 YOLO |
| `weights/players.pt` | 球员 / 门将 / 裁判 / 球 |
| `weights/craft_mlt_25k.pth` / `english_g2.pth` | 球衣号码 EasyOCR |

打开画面后点 **「球员推理」**:会加载 `weights/players.pt`(或你浏览的 `.pt`),
原图画框,俯视图用框底边中点(球用框中心)投到球场上。再点一次关闭。
点 **「号码识别」** 时 EasyOCR 从同一 `weights/` 读权重；特写里号码文案分
「未开启 / 未能识别 / #号」三种。
点原图检测框可看特写放大。

同一脚本也支持模型法推理(`--detector`):

```bash
# 本地 .pt 关键点模型
python3 scripts/run_pitch_view.py --detector yolo --model weights/pitch_keypoints.pt data/1.png
python3 scripts/run_pitch_view.py --detector yolo --model weights/pitch_keypoints.pt --gui

# Roboflow 模型(需 api key)
python3 scripts/run_pitch_view.py --detector inference data/1.png
python3 scripts/run_pitch_view.py --detector roboflow --model <model_id> data/1.png
```

## 下载 / 训练公开数据集

```bash
export ROBOFLOW_API_KEY=...

# 球场 32 关键点(YOLOv8-pose)
python3 scripts/train.py --download
python3 scripts/train.py --data-dir football-field-detection-14 \
    --epochs 100 --model yolov8m-pose.pt

# 球员 / 门将 / 裁判 / 球(YOLOv8 detect,自己训,不用现成权重)
python3 scripts/train_players.py --download --epochs 100 --model yolov8m.pt
python3 scripts/train_players.py --epochs 100 --model yolov8m.pt
# 权重自动发布到: weights/players.pt
```

选择 YOLO 时默认权重是 `runs/pose/train/weights/best.pt`（相对仓库根目录，等同 `/home/yy/pitch_view/runs/pose/train/weights/best.pt`）。`config/court_detector.yaml` 的 `model_path` 写相对路径或绝对路径都可以。球员训练细节见 [training.md](training.md) 第 7 节。

## 从转播视频抽帧并标定 32 个关键点

抽帧只保留能看到球场标线的画面，去掉特写和同一机位的重复帧。结果在
`data/pitch_dataset/images/`，清单是 `data/pitch_dataset/manifest.csv`。

```bash
python3 scripts/extract_pitch_frames.py data/master.mp4
```

标定窗口左图按文件名顺序显示这些图片，右图是带 0–31 编号的球场二维图
（顺序与 `football-field-detection-14` 一致）。窗口上方用「打开图片…」选定从哪一张开始，
用「权重…」选 `.pt`，不必在命令行里传路径。

```bash
python3 scripts/annotate_keypoints.py
```

| 操作 | 作用 |
|------|------|
| 「打开图片…」 | 选一张图，从这一张开始。同一目录的其余图片按文件名排在前后，并读取同级 `labels/` 里已有的 `.txt` |
| 「权重…」 | 换预标用的 `.pt`。默认 `weights/pitch_keypoints.pt` |
| **W** 或「权重预标」 | 用当前权重给这一张填点，再手改 |
| 拖动绿点 | 微调权重给出的位置 |
| 左键 | 当前点标为可见，并跳到下一个空位 |
| 右键 | 同一位置标为被挡住 |
| **N** | 当前点不在画面里 |
| **Z** | 撤销（预标可整张撤回） |
| **A / D** | 上一张 / 下一张 |
| 空格 | 按顺序播放，点图片即停 |
| 画面下方两行 0–31 按钮，或点击球场编号 | 跳到该点。标完后下一个仍是后一个编号，不会把中间的点标成「不在画面」 |

关掉窗口或换图时自动保存。默认图片在 `data/pitch_dataset/images/` 时，标注在
`data/pitch_dataset/labels/`（和图片同名）。`.txt` 与
`football-field-detection-14/train/labels` 相同：一行 YOLO 关键点，
`类别 cx cy w h` 接 32 组 `x y 可见性`，坐标归一化到 0–1，类别为 `pitch`（0）。
另有一份同名 `.json`，只给这个窗口记住哪些点还没标，训练不用它。
下次打开会从第一张还没标的图继续。

标完后和公开数据一起训：把图片和同名 `.txt` 放进
`football-field-detection-14/train/images` 与 `train/labels`，留一部分到
`valid/`，然后：

```bash
python3 scripts/train.py \
    --data-dir /home/yy/pitch_view/football-field-detection-14 \
    --model weights/pitch_keypoints.pt
```

点的含义和可见性（2 可见 / 1 遮挡 / 0 不在画面）见 [annotation.md](annotation.md)。

## 单图调试(关键点模型)

```bash
export ROBOFLOW_API_KEY=...
python3 scripts/dump_keypoints.py data/1.png --detector inference
```

## 运行 ROS2 节点

```bash
ros2 launch pitch_view court_detector.launch.py
```

