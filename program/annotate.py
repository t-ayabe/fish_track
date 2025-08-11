import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# モデルの読み込み（OBB対応YOLOv11モデル）
model_path = r"C:\Users\tayav\Documents\resarch\runs\obb\train\weights\best.pt"
yolo = YOLO(model_path)

# 入力動画
video_path = r"C:\Users\tayav\Documents\resarch\output_videos\mainB_recording_20250803_141847.mp4"
cap = cv2.VideoCapture(video_path)

# 出力先
output_dir = Path("annotated")
output_dir.mkdir(parents=True, exist_ok=True)

frame_idx = 0

# 検出されたクラス名を格納するセット
detected_classes = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # フレームの幅と高さを取得
    h, w, _ = frame.shape

    # 推論（OBB形式で出力される想定）
    results = yolo.predict(frame, conf=0.25, verbose=False)

    # 結果（1フレームごと）
    result = results[0]

    # クラス名をセットに追加
    if result.names:
        for box in result.obb.xywhr:
            cls_id = int(box[-1])
            if cls_id < len(result.names):
                detected_classes.add(result.names[cls_id])

    # ファイル名の準備
    img_name = f"frame_{frame_idx:05d}.jpg"
    label_name = f"frame_{frame_idx:05d}.txt"

    # 保存処理：画像
    img_path = output_dir / img_name
    cv2.imwrite(str(img_path), frame)

    # 保存処理：アノテーション（YOLO-OBB形式、正規化済み）
    label_path = output_dir / label_name
    with open(label_path, 'w') as f:
        for box, cls in zip(result.obb.xyxyxyxy, result.obb.cls):
            cls = int(cls)
            
            # .tolist() で取得した座標をフラットなリストに変換
            coords = box.tolist()
            if isinstance(coords[0], list):
                coords = [c for pair in coords for c in pair]  # ネストを解除

            # 正規化
            norm_coords = [coord / w if i % 2 == 0 else coord / h for i, coord in enumerate(coords)]

            # Roboflow形式：最初の点を最後にも追加して閉ループにする
            x1, y1 = norm_coords[0], norm_coords[1]
            norm_coords += [x1, y1]

            # 書き出し
            f.write(f"{cls} " + " ".join(f"{c:.6f}" for c in norm_coords) + "\n")



    frame_idx += 1

cap.release()

# クラスデータを classes.txt に保存
classes_file_path = output_dir / "classes.txt"
with open(classes_file_path, 'w') as f:
    for cls_name in sorted(list(detected_classes)):
        f.write(f"{cls_name}\n")

print(f"クラスデータが {classes_file_path} に保存されました。")
print(f"検出されたクラス: {list(detected_classes)}")