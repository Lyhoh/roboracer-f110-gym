#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def build_walls_from_points(npz_path_name="static_hits_raw.npz",
                            map_out_name="static_map.npz",
                            ds=0.05):
    npz_path = '/home/lyh/ros2_ws/src/f110_gym/localization/static_map/' + npz_path_name
    map_out = '/home/lyh/ros2_ws/src/f110_gym/localization/static_map/'+ map_out_name
    data = np.load(npz_path)
    wall_s = data["wall_s"]
    wall_d = data["wall_d"]
    track_length = float(data["track_length"])

    print(f"Loaded {wall_s.size} near-track points")

    s_axis = np.arange(0.0, track_length, ds)
    Ns = len(s_axis)

    d_left  = np.full(Ns, np.nan, dtype=np.float64)
    d_right = np.full(Ns, np.nan, dtype=np.float64)

    s_bins = np.concatenate([s_axis, [track_length + ds]])
    idx_bins = np.digitize(wall_s, s_bins) - 1  # 0..Ns-1

    idx_bins = np.clip(idx_bins, 0, Ns - 1)

    for i in range(Ns):
        mask_i = (idx_bins == i)
        if not np.any(mask_i):
            continue
        d_i = wall_d[mask_i]

        d_left_i = d_i[d_i > 0.0]
        if d_left_i.size > 0:
            d_left[i] = np.median(d_left_i)

        d_right_i = d_i[d_i < 0.0]
        if d_right_i.size > 0:
            d_right[i] = np.median(d_right_i)

    def smooth_nan_aware(arr, window=7):
        arr_sm = arr.copy()
        valid = ~np.isnan(arr)
        idx_valid = np.where(valid)[0]
        for idx in idx_valid:
            i0 = max(0, idx - window // 2)
            i1 = min(len(arr), idx + window // 2 + 1)
            seg = arr[i0:i1]
            seg_valid = ~np.isnan(seg)
            if np.any(seg_valid):
                arr_sm[idx] = np.mean(seg[seg_valid])
        return arr_sm

    def fill_nan_by_interp(s_axis, arr, max_gap=1.0):
        arr_f = arr.copy()
        valid = ~np.isnan(arr)
        if not np.any(valid):
            return arr_f

        s_valid = s_axis[valid]
        v_valid = arr[valid]
        v_interp = np.interp(s_axis, s_valid, v_valid)

        nearest_dist = np.abs(s_axis[:, None] - s_valid[None, :]).min(axis=1)
        use = nearest_dist <= max_gap
        arr_f[use] = v_interp[use]
        return arr_f

    d_left_sm  = smooth_nan_aware(d_left,  window=7)
    d_right_sm = smooth_nan_aware(d_right, window=7)

    d_left_f   = fill_nan_by_interp(s_axis, d_left_sm,  max_gap=1.0)
    d_right_f  = fill_nan_by_interp(s_axis, d_right_sm, max_gap=1.0)

    def limit_slope(d, max_delta=0.2):
        d_smooth = d.copy()
        for i in range(1, len(d_smooth)):
            if abs(d_smooth[i] - d_smooth[i-1]) > max_delta:
                d_smooth[i] = d_smooth[i-1]
        return d_smooth

    d_left_smooth  = limit_slope(d_left_f,  max_delta=0.2)
    d_right_smooth = limit_slope(d_right_f, max_delta=0.2)


    np.savez(map_out,
             s_axis=s_axis,
             d_left=d_left_smooth,
             d_right=d_right_smooth)

    print(f"Saved static_map to {map_out}")
    print(f"valid left: {np.sum(~np.isnan(d_left_smooth))}, "
          f"valid right: {np.sum(~np.isnan(d_right_smooth))}")

    plt.figure(figsize=(10,4))
    plt.plot(s_axis, d_left_smooth,  label="d_left")
    plt.plot(s_axis, d_right_smooth, label="d_right")
    plt.xlabel("s [m]")
    plt.ylabel("d [m]")
    plt.grid(True)
    plt.legend()
    plt.title("Fitted static walls (simple)")
    plt.show()


if __name__ == "__main__":
    build_walls_from_points()
