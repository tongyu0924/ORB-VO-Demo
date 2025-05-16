# export_trajectory.py (optimized 3D + 2D plotting with natural 3D view)
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

# Camera intrinsics
W, H, F = 960, 540, 270
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])

def normalize(K, pts):
    Kinv = np.linalg.inv(K)
    add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    norm_pts = np.dot(Kinv, add_ones(pts).T).T[:, :2]
    return norm_pts

def extract_Rt(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1
    R = U @ W @ Vt
    if np.trace(R) < 0: R = U @ W.T @ Vt
    t = U[:, 2]
    return R, t

def extract_features(img):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 3)
    keypoints = [cv2.KeyPoint(p[0][0], p[0][1], 20) for p in kps]
    keypoints, descriptors = orb.compute(gray, keypoints)
    pts = np.array([kp.pt for kp in keypoints])
    return pts, descriptors

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

# Init
cap = cv2.VideoCapture("road2.mp4")
ret, prev = cap.read()
prev_kp, prev_des = extract_features(prev)
poses = [np.eye(4)]

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Setup real-time 3D + 2D plots
plt.ion()
fig = plt.figure(figsize=(14, 5))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

ax3d.set_title("3D Trajectory (X-Z-Y)")
ax3d.set_xlabel("X (Right)")
ax3d.set_ylabel("Z (Forward)")
ax3d.set_zlabel("Y (Up)")
ax3d.view_init(elev=30, azim=-120)  # natural 3D viewing angle

ax2d.set_title("2D Trajectory (X-Z)")
ax2d.set_xlabel("X (Right)")
ax2d.set_ylabel("Z (Forward)")

xs, ys, zs = [], [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    curr_kp, curr_des = extract_features(frame)
    matches = bf.knnMatch(curr_des, prev_des, k=2)
    good = [(m.trainIdx, m.queryIdx) for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8: continue

    matched_prev = np.array([prev_kp[i] for i, _ in good])
    matched_curr = np.array([curr_kp[j] for _, j in good])

    norm_prev = normalize(K, matched_prev)
    norm_curr = normalize(K, matched_curr)
    model, inliers = ransac((norm_prev, norm_curr), EssentialMatrixTransform, min_samples=8,
                            residual_threshold=0.005, max_trials=100)

    E = model.params
    R, t = extract_Rt(E)
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t

    poses.append(poses[-1] @ np.linalg.inv(Rt))
    prev_kp, prev_des = curr_kp, curr_des

    pose_t = poses[-1][:3, 3]
    xs.append(pose_t[0])
    ys.append(pose_t[1])
    zs.append(pose_t[2])

    ax3d.clear()
    ax3d.plot(xs, zs, ys, c='blue', label='Path')
    ax3d.scatter(xs[0], zs[0], ys[0], c='green', label='Start')
    ax3d.scatter(xs[-1], zs[-1], ys[-1], c='red', label='Now')
    ax3d.set_title("3D Trajectory (X-Z-Y)")
    ax3d.set_xlabel("X (Right)")
    ax3d.set_ylabel("Z (Forward)")
    ax3d.set_zlabel("Y (Up)")
    ax3d.view_init(elev=30, azim=-120)
    ax3d.legend()
    set_axes_equal(ax3d)

    ax2d.clear()
    ax2d.plot(xs, zs, c='blue', label='X-Z Path')
    ax2d.scatter(xs[0], zs[0], c='green', label='Start')
    ax2d.scatter(xs[-1], zs[-1], c='red', label='Now')
    ax2d.set_title("2D Trajectory (X-Z)")
    ax2d.set_xlabel("X (Right)")
    ax2d.set_ylabel("Z (Forward)")
    ax2d.axis('equal')
    ax2d.legend()

    plt.pause(0.01)

cap.release()

# Save JSON
output = [[x, y, z] for x, y, z in zip(xs, ys, zs)]
Path("trajectory.json").write_text(json.dumps(output, indent=2))
print(f"Exported {len(output)} poses to trajectory.json")

plt.ioff()
plt.show()

