import cv2
from ultralytics import YOLO

# 学習済みのモデルをロード
# model = YOLO('yolo11n.pt')
model = YOLO(r'C:\Users\tayav\Documents\resarch\runs\obb\train\weights\best.pt')

# 動画ファイルのパスを指定
file_name = 'fish_test2'
video_path = "./test_movie/" + file_name + ".mp4"
cap = cv2.VideoCapture(video_path)


# 出力ファイルの設定
output_path = "./val_movie/" + file_name + ".mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力ファイル形式
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print('pass')
# キーが押されるまでループ
while cap.isOpened():
    # １フレーム読み込む
    success, frame = cap.read()

    # rotate the frame
    # frame = cv2.rotate(frame, cv2.ROTATE_180)

    if success:
        # YOLOv12でトラッキング
        results = model.track(frame, persist=True)

        # 結果を画像に変換
        annotated_frame = results[0].plot()

        # フレームを出力ファイルに書き込む
        out.write(annotated_frame)

        # OpenCVで表示＆キー入力チェック
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        key = cv2.waitKey(1)
        if key != -1:
            print("STOP PLAY")
            break
    else:
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()