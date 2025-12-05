# import numpy as np
# import matplotlib.pyplot as plt

# # 读取CSV，假设有表头：x_m,y_m,s_m,d_left,d_right
# data = np.genfromtxt('/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_centerline.csv', delimiter=",", names=True)

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

# import numpy as np
# import matplotlib.pyplot as plt

# # 读取CSV
# data = np.genfromtxt('/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_centerline.csv', delimiter=",", names=True)

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

# # --- 绘制散点 ---
# plt.figure(figsize=(8, 6))
# plt.scatter(centers[:,0], centers[:,1], c='k', s=10, label="Center")
# plt.scatter(left_pts[:,0], left_pts[:,1], c='r', s=10, label="Left boundary")
# plt.scatter(right_pts[:,0], right_pts[:,1], c='b', s=10, label="Right boundary")

# plt.axis('equal')
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.legend()
# plt.title("Track points (scatter)")
# plt.show()

# a = set([1, 2, 3])
# a = set()
# b = set([3, 4, 5])
# print(a - b) 

# reorder_waypoints_and_recompute_s.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path

# ======= CONFIG （直接改这里即可）=======
INPUT_CSV  = Path("/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_centerline.csv")                 # 原始CSV
OUTPUT_CSV = Path("/home/lyh/ros2_ws/src/f110_gym/perception/waypoints/map5/global_centerline1.csv")       # 输出CSV
START_ROW_WITH_HEADER = 257  # 以整表行号计：表头=第1行 -> 从第257行数据开始
ADD_WRAP_DISTANCE_TO_LAST = False  # 若想让最后一个点的s包含“末点到首点”的闭环距离，设为True
# =======================================

def main():
    df = pd.read_csv(INPUT_CSV)

    # 备份原始 s 列，便于对比
    if 's_m' in df.columns:
        df.insert(df.columns.get_loc('s_m') + 1, 's_m_orig', df['s_m'])
    else:
        raise ValueError("CSV中缺少列 's_m'。")

    # 计算数据区起始索引（0-based）
    start_idx = START_ROW_WITH_HEADER - 2  # 表头=1行，数据第一行=第2行 -> 减2
    if start_idx < 0:
        raise ValueError("START_ROW_WITH_HEADER 必须 ≥ 2（第1行为表头）。")
    if start_idx >= len(df):
        raise ValueError(
            f"START_ROW_WITH_HEADER={START_ROW_WITH_HEADER} 超出数据范围。"
            f"数据行数（不含表头）为 {len(df)}，最大可取 {len(df)+1}。"
        )

    # 环形重排
    df = pd.concat([df.iloc[start_idx:], df.iloc[:start_idx]], ignore_index=True)

    # 重新计算 s_m：以 (x_m, y_m) 的相邻欧氏距离累计，s[0] = 0
    if not {'x_m', 'y_m'}.issubset(df.columns):
        raise ValueError("CSV中缺少列 'x_m' 或 'y_m'。")

    dx = df['x_m'].diff()
    dy = df['y_m'].diff()
    seg = np.hypot(dx, dy).fillna(0.0)     # 第一项为 0
    s_new = seg.cumsum()                   # 从0累加到最后

    if ADD_WRAP_DISTANCE_TO_LAST and len(df) > 1:
        wrap = float(np.hypot(
            df['x_m'].iloc[-1] - df['x_m'].iloc[0],
            df['y_m'].iloc[-1] - df['y_m'].iloc[0],
        ))
        s_new.iloc[-1] += wrap

    df['s_m'] = s_new.values

    # 导出
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Wrote -> {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    main()