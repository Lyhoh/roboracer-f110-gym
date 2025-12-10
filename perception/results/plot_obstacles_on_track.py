import numpy as np
import matplotlib.pyplot as plt
from roboracer_utils.frenet_converter import FrenetConverter


def project_point_to_polyline(px, py, poly):
    """
    Project point (px, py) onto a polyline poly (Nx2 array).
    Return (min_dist, proj_x, proj_y).
    """
    if poly.shape[0] < 2:
        return np.inf, np.nan, np.nan

    # segments: P0->P1, P1->P2, ...
    x0 = poly[:-1, 0]
    y0 = poly[:-1, 1]
    x1 = poly[1:, 0]
    y1 = poly[1:, 1]

    dx = x1 - x0
    dy = y1 - y0
    seg_len2 = dx*dx + dy*dy + 1e-8

    # projection parameter t on each segment
    t = ((px - x0)*dx + (py - y0)*dy) / seg_len2
    t = np.clip(t, 0.0, 1.0)

    proj_x = x0 + t * dx
    proj_y = y0 + t * dy

    dist2 = (proj_x - px)**2 + (proj_y - py)**2
    idx = np.argmin(dist2)

    return float(np.sqrt(dist2[idx])), float(proj_x[idx]), float(proj_y[idx])


def main(npz_path="/home/lyh/ros2_ws/src/f110_gym/perception/results/obstacles_debug.npz",
         static_map_path="/home/lyh/ros2_ws/src/f110_gym/localization/static_map/static_map.npz",
         d_vis_max=3.0):
    data = np.load(npz_path)
    obs_x = data["obs_x"]
    obs_y = data["obs_y"]
    obs_s = data["obs_s"]
    obs_d = data["obs_d"]
    track_x = data["track_x"]
    track_y = data["track_y"]
    left_x = data["left_x"]
    left_y = data["left_y"]
    right_x = data["right_x"]
    right_y = data["right_y"]
    track_length = float(data["track_length"])
    left_wall = np.column_stack([left_x, left_y])
    right_wall = np.column_stack([right_x, right_y])

    cls_s = data["cls_s"]
    cls_d = data["cls_d"]
    cls_kept = data["cls_kept"]
    cls_x = data["cls_x"]
    cls_y = data["cls_y"]
    center_x = data["center_x"]
    center_y = data["center_y"]
    geom_x   = data["geom_x"]
    geom_y   = data["geom_y"]
    st_x     = data["st_x"]
    st_y     = data["st_y"]


    if cls_d.size > 0:
        # mask_cls = np.abs(cls_d) < d_vis_max
        mask_cls = (cls_kept == 1) & (np.abs(cls_d) < d_vis_max)
        cls_x_vis = cls_x[mask_cls]
        cls_y_vis = cls_y[mask_cls]
        cls_s_vis = cls_s[mask_cls]
        cls_d_vis = cls_d[mask_cls]
        # cls_kept_vis = cls_kept[mask_cls]

        center_x_vis = center_x[mask_cls]
        center_y_vis = center_y[mask_cls]

        geom_x_vis   = geom_x[mask_cls]
        geom_y_vis   = geom_y[mask_cls]

        st_x_vis     = st_x[mask_cls]
        st_y_vis     = st_y[mask_cls]
    else:
        cls_x_vis = cls_y_vis = cls_s_vis = cls_d_vis = cls_kept_vis = cls_d

    wall_px_vis = []
    wall_py_vis = []
    wall_dist_vis = []


    for x_c, y_c in zip(cls_x_vis, cls_y_vis):
        # 投影到左墙
        dL, pxL, pyL = project_point_to_polyline(x_c, y_c, left_wall)
        # 投影到右墙
        dR, pxR, pyR = project_point_to_polyline(x_c, y_c, right_wall)

        # 选最近的一侧
        if dL < dR:
            wall_px_vis.append(pxL)
            wall_py_vis.append(pyL)
            wall_dist_vis.append(dL)
        else:
            wall_px_vis.append(pxR)
            wall_py_vis.append(pyR)
            wall_dist_vis.append(dR)

    wall_px_vis = np.array(wall_px_vis, dtype=float)
    wall_py_vis = np.array(wall_py_vis, dtype=float)
    wall_dist_vis = np.array(wall_dist_vis, dtype=float)

    print(f"Loaded {obs_x.size} obstacle samples")
    print(f"Track points: center={track_x.size}, left={left_x.size}, right={right_x.size}")

    # try to load static walls (optional)
    try:
        smap = np.load(static_map_path)
        s_axis = smap["s_axis"]
        d_left = smap["d_left"]
        d_right = smap["d_right"]
        has_static = True
    except Exception as e:
        print(f"Static map not loaded: {e}")
        has_static = False

    if track_x.size > 0:
        converter = FrenetConverter(track_x.astype(np.float64),
                                    track_y.astype(np.float64))
    else:
        converter = None

        # ==== 计算 Frenet d: 障碍物 vs 对应最近墙点 ====
    if converter is not None and cls_x_vis.size > 0:
        # 障碍物点的 Frenet
        s_obs, d_obs = converter.get_frenet(
            cls_x_vis.astype(np.float64),
            cls_y_vis.astype(np.float64)
        )
        s_obs = s_obs.astype(float)
        d_obs = d_obs.astype(float)

        # 墙最近点的 Frenet
        s_wall, d_wall = converter.get_frenet(
            wall_px_vis.astype(np.float64),
            wall_py_vis.astype(np.float64)
        )
        s_wall = s_wall.astype(float)
        d_wall = d_wall.astype(float)
    else:
        s_obs = d_obs = s_wall = d_wall = np.array([], dtype=float)

    if d_obs.size > 0:
        print("\n==== Compare |d_obstacle| vs |d_wall| (Frenet) ====")
        for i in range(len(d_obs)):
            d_o = d_obs[i]
            d_w = d_wall[i]
            inside = abs(d_o) < abs(d_w)   # 是否在墙内侧
            side = "inside" if inside else "outside"
            print(f"[#{i:03d}] d_obs={d_o:+.3f}, d_wall={d_w:+.3f}, "
                  f"|d_obs| {'<' if abs(d_o)<abs(d_w) else '>'} |d_wall|  -> {side}")


    # ---------- Figure 1: map frame (x,y) ----------
    plt.figure(figsize=(10, 5))

    # centerline
    if track_x.size > 0:
        plt.plot(track_x, track_y, 'k-', label="centerline")

    # track boundaries
    if left_x.size > 0:
        plt.plot(left_x, left_y, 'g-', label="left boundary")
    if right_x.size > 0:
        plt.plot(right_x, right_y, 'b-', label="right boundary")

    # obstacles
    if obs_x.size > 0:
        plt.scatter(obs_x, obs_y, s=8, alpha=0.7, c='r', label="detected obstacles")

    # cluster representative points (outermost points used for wall check)
    if cls_x_vis.size > 0:
        # kept_mask = (cls_kept_vis == 1)
        # filt_mask = (cls_kept_vis == 0)

        # if np.any(kept_mask):
        #     plt.scatter(cls_x_vis[kept_mask], cls_y_vis[kept_mask],
        #                 s=10, alpha=0.7, c='orange',
        #                 marker='x', label="cluster rep kept (|d|<%.1f)" % d_vis_max)
            
        #     if converter is not None:
        #         step = 1  # 想稀疏一点可以改大，比如 3、5
        #         for i in range(0, cls_s_vis[kept_mask].size, step):
        #             s_c = float(cls_s_vis[kept_mask][i]) % track_length
        #             d_c = float(cls_d_vis[kept_mask][i])
        #             x_c = float(cls_x_vis[kept_mask][i])
        #             y_c = float(cls_y_vis[kept_mask][i])

        #             x_axis, y_axis = converter.get_cartesian(s_c, 0.0)
        #             x_axis = float(x_axis); y_axis = float(y_axis)

        #             plt.plot([x_c, x_axis], [y_c, y_axis],
        #                     linewidth=0.6, alpha=0.5, color='magenta')
                    
        #             # === 连到“用于比较的墙点” ===
        #             # 1) 按 detect.py 里的 is_static_background 逻辑找最近的 s 索引
        #             idx = np.searchsorted(s_axis, s_c) - 1
        #             idx = max(0, min(idx, len(s_axis) - 1))

        #             # 2) 根据 d 的正负选择左/右墙
        #             if d_c >= 0:
        #                 d_wall = float(d_left[idx])
        #             else:
        #                 d_wall = float(d_right[idx])

        #             # 3) 有墙值就转回 (x,y) 并画线
        #             if not np.isnan(d_wall):
        #                 s_wall = float(s_axis[idx])
        #                 x_wall, y_wall = converter.get_cartesian(s_wall, d_wall)
        #                 plt.plot([x_c, x_wall], [y_c, y_wall],
        #                         linewidth=0.6, alpha=0.5, color='lime')

        # for i in range(cls_x_vis.size):
        #     d_left_xy = point_to_polyline_distance(cls_x_vis[i], cls_y_vis[i], left_wall)
        #     d_right_xy = point_to_polyline_distance(cls_x_vis[i], cls_y_vis[i], right_wall)
        #     d_wall_xy = min(d_left_xy, d_right_xy)

        # colors = np.clip(d_wall_xy, 0, 0.5)   # 0~0.5 m 映射颜色
        # plt.scatter(cls_x_vis, cls_y_vis, c=colors, cmap='jet', s=20)
        # plt.colorbar(label="distance to wall (m)")
        plt.scatter(cls_x_vis, cls_y_vis,
                    s=14, alpha=0.9, c='orange',
                    marker='x', label="kept cluster rep (|d|<%.1f)" % d_vis_max)

        # 画障碍点 → 最近墙的连线（基于 x,y 最近距离，不用 Frenet）
        for x_c, y_c, wx, wy in zip(cls_x_vis, cls_y_vis, wall_px_vis, wall_py_vis):
            plt.plot([x_c, wx], [y_c, wy],
                     linewidth=0.6, alpha=0.5, color='cyan')


        #     # 画障碍物真实点
        # plt.scatter(cls_x_vis, cls_y_vis,
        #             s=10, alpha=0.8, c='orange', marker='x', label="kept cluster rep")

        # # === 画三种连线 ===
        # for x_c, y_c, xc, yc, xg, yg, xs, ys in zip(
        #         cls_x_vis, cls_y_vis,
        #         center_x_v    qis, center_y_vis,
        #         geom_x_vis,   geom_y_vis,
        #         st_x_vis,     st_y_vis):

        #     # → centerline
        #     plt.plot([x_c, xc], [y_c, yc],
        #             linewidth=0.7, alpha=0.6, color='magenta')

        #     # → geometric wall
        #     plt.plot([x_c, xg], [y_c, yg],
        #             linewidth=0.7, alpha=0.6, color='orange')

        #     # → static wall
        #     if not np.isnan(xs):
        #         plt.plot([x_c, xs], [y_c, ys],
        #                 linewidth=0.7, alpha=0.6, color='cyan')

        # if np.any(filt_mask):
        #     plt.scatter(cls_x_vis[filt_mask], cls_y_vis[filt_mask],
        #                 s=10, alpha=0.4, c='cyan',
        #                 marker='x', label="cluster rep filtered (|d|<%.1f)" % d_vis_max)


    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Track + boundaries + detected obstacles (map frame)")
    plt.grid(True)
    plt.savefig("/home/lyh/ros2_ws/src/f110_gym/perception/results/map_debug_xy.png", dpi=200, bbox_inches="tight")
    plt.legend()

    # ---------- Figure 2: Frenet (s,d) ----------
    plt.figure(figsize=(10, 5))
    if has_static:
        plt.plot(s_axis, d_left,  'g-', label="static d_left")
        plt.plot(s_axis, d_right, 'b-', label="static d_right")

    # if cls_s_vis.size > 0:
    #     kept_mask = (cls_kept_vis == 1)
    #     filt_mask = (cls_kept_vis == 0)

    #     plt.scatter(cls_s_vis[kept_mask], cls_d_vis[kept_mask],
    #                 s=8, c='r', alpha=0.7, label="clusters kept (|d|<%.1f)" % d_vis_max)

    #     plt.scatter(cls_s_vis[filt_mask], cls_d_vis[filt_mask],
    #                 s=8, c='gray', alpha=0.5, label="clusters filtered (|d|<%.1f)" % d_vis_max)
    if cls_s_vis.size > 0:
        plt.scatter(cls_s_vis, cls_d_vis,
                    s=8, c='r', alpha=0.7, label="clusters kept (|d|<%.1f)" % d_vis_max)

    plt.xlabel("s [m]")
    plt.ylabel("d [m]")
    plt.title("Cluster-level (s,d) vs static walls")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/lyh/ros2_ws/src/f110_gym/perception/results/map_debug_sd.png", dpi=200, bbox_inches="tight")
    plt.show()

    


if __name__ == "__main__":
    main()
