# 用法

统一配置见仓库根目录 [`config/court_detector.yaml`](../config/court_detector.yaml)
（球员置信度、NMS、OCR、特写等均可改；GUI / CLI / ROS 共用）。

```bash
python3 scripts/run_homography.py --config config/court_detector.yaml
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
python3 scripts/run_homography.py --calibration data/1_calibration.json data/1.png

# 视频 -> 生成 *_warped.mp4
python3 scripts/run_homography.py --calibration calib.json --video match.mp4 \
    --scale 0.05 --out-dir out/

# Qt 窗口:可选输入源(文件/相机/RTSP),左原图右俯视图
python3 scripts/run_homography.py --calibration calib.json
```

推理权重统一放在 `models/`（打包带走即可）:

| 文件 | 用途 |
|------|------|
| `models/pitch_keypoints.pt` | 球场关键点 YOLO |
| `models/players.pt` | 球员 / 门将 / 裁判 / 球 |
| `models/craft_mlt_25k.pth` / `english_g2.pth` | 球衣号码 EasyOCR |

打开画面后点 **「球员推理」**:会加载 `models/players.pt`(或你浏览的 `.pt`),
原图画框,俯视图用框底边中点(球用框中心)投到球场上。再点一次关闭。
点 **「号码识别」** 时 EasyOCR 从同一 `models/` 读权重；特写里号码文案分
「未开启 / 未能识别 / #号」三种。
点原图检测框可看特写放大。

同一脚本也支持模型法推理(`--detector`):

```bash
# 本地 .pt 关键点模型
python3 scripts/run_homography.py --detector yolo --model models/pitch_keypoints.pt data/1.png
python3 scripts/run_homography.py --detector yolo --model models/pitch_keypoints.pt --gui

# Roboflow 模型(需 api key)
python3 scripts/run_homography.py --detector inference data/1.png
python3 scripts/run_homography.py --detector roboflow --model <model_id> data/1.png
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
# 权重自动发布到: models/players.pt
```

关键点训好后权重在 `models/pitch_keypoints.pt`；`config/court_detector.yaml` 已默认
`model_path: "models/pitch_keypoints.pt"`。球员训练细节见 [training.md](training.md) 第 7 节。

## 单图调试(关键点模型)

```bash
export ROBOFLOW_API_KEY=...
python3 scripts/dump_keypoints.py data/1.png --detector inference
```

## 运行 ROS2 节点

```bash
ros2 launch pitch_view court_detector.launch.py
```

