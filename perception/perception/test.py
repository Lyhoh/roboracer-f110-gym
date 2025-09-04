# import numpy as np
# import matplotlib.pyplot as plt

# # 读取CSV，假设有表头：x_m,y_m,s_m,d_left,d_right
# data = np.genfromtxt('/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_waypoints.csv', delimiter=",", names=True)

# x = data['x_m']
# y = data['y_m']
# s = data['s_m']
# dL = data['d_left']
# dR = data['d_right']

# # --- 计算切向量和法向量 ---
# dxds = np.gradient(x, s)
# dyds = np.gradient(y, s)
# T = np.stack([dxds, dyds], axis=1)
# T_norm = np.linalg.norm(T, axis=1, keepdims=True)
# T_norm[T_norm == 0] = 1.0
# T = T / T_norm
# N = np.stack([-T[:,1], T[:,0]], axis=1)  # 左法向量

# centers = np.stack([x, y], axis=1)
# left_pts  = centers + N * dL[:, None]
# right_pts = centers - N * dR[:, None]

# # --- 绘图 ---
# plt.figure(figsize=(8, 6))
# # 中心线
# plt.plot(x, y, 'k-', label="Centerline")
# # 左右边界
# plt.plot(left_pts[:,0], left_pts[:,1], 'r-', label="Left boundary")
# plt.plot(right_pts[:,0], right_pts[:,1], 'b-', label="Right boundary")
# # 可选：画出部分法线
# for i in range(0, len(x), max(1, len(x)//30)):  # 每隔若干点画一条
#     plt.plot([centers[i,0], left_pts[i,0]], [centers[i,1], left_pts[i,1]], 'r--', lw=0.5)
#     plt.plot([centers[i,0], right_pts[i,0]], [centers[i,1], right_pts[i,1]], 'b--', lw=0.5)

# plt.axis('equal')
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.legend()
# plt.title("Track visualization from CSV")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 读取CSV
data = np.genfromtxt('/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_waypoints.csv', delimiter=",", names=True)

x = data['x_m']
y = data['y_m']
s = data['s_m']
dL = data['d_left']
dR = data['d_right']

# --- 计算切向量和法向量 ---
dxds = np.gradient(x, s)
dyds = np.gradient(y, s)
T = np.stack([dxds, dyds], axis=1)
T_norm = np.linalg.norm(T, axis=1, keepdims=True)
T_norm[T_norm == 0] = 1.0
T = T / T_norm
N = np.stack([-T[:,1], T[:,0]], axis=1)  # 左法向量

centers = np.stack([x, y], axis=1)
left_pts  = centers + N * dL[:, None]
right_pts = centers - N * dR[:, None]

# --- 绘制散点 ---
plt.figure(figsize=(8, 6))
plt.scatter(centers[:,0], centers[:,1], c='k', s=10, label="Center")
plt.scatter(left_pts[:,0], left_pts[:,1], c='r', s=10, label="Left boundary")
plt.scatter(right_pts[:,0], right_pts[:,1], c='b', s=10, label="Right boundary")

plt.axis('equal')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.title("Track points (scatter)")
plt.show()