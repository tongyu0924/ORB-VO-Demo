# ORB-VO-Demo

🎬 **Demo Video**  
▶️ [Watch on YouTube](https://youtu.be/LPUv11dxp4c)


![Demo Screenshot](./screenshot.png)

```
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
```
