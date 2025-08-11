import cv2
import numpy as np
import os
import datetime

# ===== 実験パラメータ =====
n_air = 1.0003
n_glass = 1.50
n_water = 1.333
t_glass_front = 10.0   # mm
t_water = 300.0        # mm
t_glass_back = 10.0    # mm

# チェスボード設定
chessboard_size = (26, 14)   # 交点数
square_size = 15.0         # mm

# 保存先
save_dir = "calibration_images"
os.makedirs(save_dir, exist_ok=True)

# ===== 屈折計算関数 =====
def refract_vector(v, n, n1, n2):
    cos_theta_i = -np.dot(n, v)
    sin2_theta_t = (n1 / n2)**2 * (1 - cos_theta_i**2)
    if sin2_theta_t > 1.0:
        return None
    cos_theta_t = np.sqrt(1 - sin2_theta_t)
    return (n1/n2) * v + (n1/n2 * cos_theta_i - cos_theta_t) * n

def trace_ray_to_monitor(pixel, mtx):
    fx, fy = mtx[0,0], mtx[1,1]
    cx, cy = mtx[0,2], mtx[1,2]
    
    xn = (pixel[0] - cx) / fx
    yn = (pixel[1] - cy) / fy
    ray = np.array([xn, yn, 1.0])
    ray /= np.linalg.norm(ray)

    n = np.array([0, 0, -1.0])

    ray_g1 = refract_vector(ray, n, n_air, n_glass)
    if ray_g1 is None: return None
    p_exit_g1 = ray_g1 * (t_glass_front / ray_g1[2])

    ray_w = refract_vector(ray_g1, n, n_glass, n_water)
    if ray_w is None: return None
    p_exit_w = p_exit_g1 + ray_w * (t_water / ray_w[2])

    ray_g2 = refract_vector(ray_w, n, n_water, n_glass)
    if ray_g2 is None: return None
    p_exit_g2 = p_exit_w + ray_g2 * (t_glass_back / ray_g2[2])

    ray_air2 = refract_vector(ray_g2, n, n_glass, n_air)
    if ray_air2 is None: return None

    return p_exit_g2[:2]

# ===== 対話型撮影 =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

print("\n--- キャリブレーション撮影開始 ---")
print("チェスボードをモニター背面（水槽部分）に表示してください")
print("スペースキー: 撮影保存 / qキー: 撮影終了\n")

img_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    display = frame.copy()
    if ret_cb:
        cv2.drawChessboardCorners(display, chessboard_size, corners, ret_cb)

    cv2.imshow("Calibration Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = os.path.join(save_dir, f"calib_{img_count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"保存: {filename}")
        img_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===== キャリブレーション計算 =====
print("\n--- キャリブレーション計算開始 ---")

objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

obj_points = []
img_points = []

image_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".jpg")]

for fname in image_files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
print("カメラ行列:\n", mtx)
print("歪み係数:\n", dist)

# 屈折補正
corrected_points = []
for img_pts in img_points:
    corrected = []
    for pt in img_pts:
        p_monitor = trace_ray_to_monitor(pt[0], mtx)
        if p_monitor is not None:
            corrected.append(p_monitor)
    corrected_points.append(np.array(corrected, dtype=np.float32))

monitor_coords = objp[:, :2].astype(np.float32)
H, _ = cv2.findHomography(monitor_coords, corrected_points[0])
print("ホモグラフィー行列:\n", H)

# 保存
np.save("camera_matrix.npy", mtx)
np.save("dist_coeffs.npy", dist)
np.save("perspective_matrix.npy", H)
print("\n--- キャリブレーション完了 ---")
print("camera_matrix.npy, dist_coeffs.npy, perspective_matrix.npy を保存しました")