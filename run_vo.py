import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

class Map:
    def __init__(self):
        self.poses = []
        self.points = []

    def add_observation(self, pose, points):
        self.poses.append(pose[:3, 3])
        self.points.extend(points[:, :3])

    def display(self):
        poses = np.array(self.poses)
        points = np.array(self.points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Camera Trajectory and 3D Points")

        if len(poses) > 0:
            ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], c='blue', label='Camera trajectory')
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', s=1, label='3D Points')

        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

def normalize(K, pts):
    Kinv = np.linalg.inv(K)
    add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    norm_pts = np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
    return norm_pts

class Frame:
    idx = 0
    last_kps, last_des, last_pose = None, None, None

    def __init__(self, image):
        Frame.idx += 1
        self.image = image
        self.idx = Frame.idx
        self.last_kps = Frame.last_kps
        self.last_des = Frame.last_des
        self.last_pose = Frame.last_pose

def extract_points(frame):
    orb = cv2.ORB_create()
    image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(image, 3000, 0.01, 3)
    kps = [cv2.KeyPoint(pt[0][0], pt[0][1], 20) for pt in pts]
    kps, des = orb.compute(image, kps)
    kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
    return kps, des

def match_points(frame):
    bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bfmatch.knnMatch(frame.curr_des, frame.last_des, k=2)
    match_kps = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            curr_pt = frame.curr_kps[m.queryIdx]
            last_pt = frame.last_kps[m.trainIdx]
            match_kps.append((curr_pt, last_pt))
    return match_kps

def fit_essential_matrix(match_kps):
    global K
    match_kps = np.array(match_kps)
    norm_curr_kps = normalize(K, match_kps[:, 0])
    norm_last_kps = normalize(K, match_kps[:, 1])
    model, inliers = ransac((norm_last_kps, norm_curr_kps), EssentialMatrixTransform,
                            min_samples=8, residual_threshold=0.005, max_trials=200)
    return model.params, match_kps[inliers]

def extract_Rt(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1
    R = np.dot(U, np.dot(W, Vt))
    if np.sum(R.diagonal()) < 0:
        R = np.dot(U, np.dot(W.T, Vt))
    t = U[:, 2]
    if t[2] < 0:
        t *= -1
    Rt = np.eye(4)
    Rt[:3, :3], Rt[:3, 3] = R, t
    return Rt

def triangulate(pts1, pts2, pose1, pose2):
    global K
    pose1, pose2 = np.linalg.inv(pose1), np.linalg.inv(pose2)
    pts1, pts2 = normalize(K, pts1), normalize(K, pts2)
    points4d = np.zeros((pts1.shape[0], 4))
    for i, (kp1, kp2) in enumerate(zip(pts1, pts2)):
        A = np.vstack([
            kp1[0] * pose1[2] - pose1[0], kp1[1] * pose1[2] - pose1[1],
            kp2[0] * pose2[2] - pose2[0], kp2[1] * pose2[2] - pose2[1]
        ])
        _, _, vt = np.linalg.svd(A)
        points4d[i] = vt[3]
    points4d /= points4d[:, [3]]
    return points4d

def draw_points(frame, match_kps):
    for kp1, kp2 in match_kps:
        u1, v1, u2, v2 = int(kp1[0]), int(kp1[1]), int(kp2[0]), int(kp2[1])
        cv2.circle(frame.image, (u1, v1), 3, (0,0,255), -1)
        cv2.line(frame.image, (u1, v1), (u2, v2), (255,0,0), 1)

def check_points(points4d):
    return points4d[:, 2] > 0

def process_frame(frame):
    frame.curr_kps, frame.curr_des = extract_points(frame)
    Frame.last_kps, Frame.last_des = frame.curr_kps, frame.curr_des
    if frame.idx == 1:
        frame.curr_pose = np.eye(4)
        points4d = np.array([[0, 0, 0, 1]])
    else:
        match_kps = match_points(frame)
        E, filtered_kps = fit_essential_matrix(match_kps)
        Rt = extract_Rt(E)
        frame.curr_pose = Rt @ frame.last_pose
        pts1 = np.array([pair[1] for pair in filtered_kps])
        pts2 = np.array([pair[0] for pair in filtered_kps])
        points4d = triangulate(pts1, pts2, frame.last_pose, frame.curr_pose)
        points4d = points4d[check_points(points4d)]
        draw_points(frame, filtered_kps)
    mapp.add_observation(frame.curr_pose, points4d)
    Frame.last_pose = frame.curr_pose
    return frame

if __name__ == "__main__":
    W, H, F = 960, 540, 270
    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
    mapp = Map()
    cap = cv2.VideoCapture("road2.mp4")

    while cap.isOpened():
        ret, image = cap.read()
        if not ret: break

        frame = process_frame(Frame(image))
        cv2.imshow("slam", frame.image)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    mapp.display()
