# ORB-based Visual Odometry (Monocular VO)

This project demonstrates a simple **camera-based localization** system using **ORB feature tracking** and **Essential matrix estimation** for monocular visual odometry.

<p align="center">
  <img src="https://img.youtube.com/vi/LPUv11dxp4c/0.jpg" width="420"/>
  <img src="https://img.youtube.com/vi/Dr8zx3VXZBE/0.jpg" width="420"/>
</p>

---

## 🎬 Demo Videos

- 🔍 **ORB Feature Tracking**  
  [▶ Watch on YouTube](https://youtu.be/LPUv11dxp4c)

- 📍 **3D/2D Trajectory Visualization**  
  [▶ Watch on YouTube](https://youtu.be/Dr8zx3VXZBE)

---

## ✨ Features

- ORB feature detection & matching
- Essential matrix (E) estimation with RANSAC
- Relative camera pose tracking
- 3D + 2D trajectory live plotting

---

## 🚀 How to Run

```bash
python export_trajectory.py


---

## 🔁 Pipeline Overview

```text
+---------------------------+
|   Load video frame       |
+---------------------------+
            |
            v
+---------------------------+
|  Convert to grayscale     |
|  Detect corners (GFTT)    |
+---------------------------+
            |
            v
+---------------------------+
|  Compute ORB descriptors  |
+---------------------------+
            |
            v
+---------------------------+
|  Match ORB features       |
+---------------------------+
            |
            v
+---------------------------+
| Estimate Essential Matrix |
|         via RANSAC        |
+---------------------------+
            |
            v
+---------------------------+
| Decompose E to R and t    |
+---------------------------+
            |
            v
+---------------------------+
|   Triangulate 3D points   |
+---------------------------+
            |
            v
+---------------------------+
|  Filter valid 3D points   |
+---------------------------+
            |
            v
+---------------------------+
| Update map and trajectory |
+---------------------------+
            |
            v
+---------------------------+
| Draw matches on frame     |
+---------------------------+
            |
            v
+---------------------------+
| Show frame / write video  |
+---------------------------+

