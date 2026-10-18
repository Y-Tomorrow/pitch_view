# 推理权重（`weights/`）

训练与推理共用本目录，路径是仓库根目录下的 `weights/`。打包部署时整夹带走即可。

| 文件 | 用途 | 写入方 |
|------|------|--------|
| `pitch_keypoints.pt` | 球场关键点 YOLO | `scripts/train.py` 训完自动发布 |
| `players.pt` | 球员 / 门将 / 裁判 / 球 | `scripts/train_players.py` 训完自动发布 |
| `craft_mlt_25k.pth` | EasyOCR 文字检测 | 首次号码识别时下载，或手动放入 |
| `english_g2.pth` | EasyOCR 英文识别 | 同上 |
| `paddle_ocr/` | PaddleOCR 英文数字模型缓存 | 界面选「PaddleOCR 数字」后首次下载 |

`runs/` 只保留 Ultralytics 训练日志；正式权重以这里为准。
路径常量见 `pitch_view/common/weights.py`。
