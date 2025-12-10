from typing import Union
import numpy as np
from scipy.interpolate import CubicSpline

class FrenetConverter:
    def __init__(self, waypoints_x: np.array, waypoints_y: np.array, waypoints_psi: np.array = None):
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y
        self.waypoints_psi = waypoints_psi
        self.waypoints_s = None
        self.spline_x = None
        self.spline_y = None
        self.raceline_length = None

        self.waypoints_distance_m = None 

        self.iter_max = 3

        self.build_raceline()

    def build_raceline(self):
        self.waypoints_s = [0.0]
        prev_wpnt_x = self.waypoints_x[0]
        prev_wpnt_y = self.waypoints_y[0]
        for wpnt_x, wpnt_y in zip(self.waypoints_x[1:], self.waypoints_y[1:]):
            dist = np.linalg.norm([wpnt_x - prev_wpnt_x, wpnt_y - prev_wpnt_y])
            prev_wpnt_x = wpnt_x
            prev_wpnt_y = wpnt_y
            self.waypoints_s.append(self.waypoints_s[-1] + dist)
               
        self.waypoints_s = np.asarray(self.waypoints_s, dtype=float)

        if len(self.waypoints_s) > 1:
            mean_ds = np.mean(np.diff(self.waypoints_s))
        else:
            mean_ds = 0.0
        self.waypoints_distance_m = mean_ds   

        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        self.raceline_length = float(self.waypoints_s[-1])

    def _s_to_index(self, s: Union[float, np.ndarray]) -> np.ndarray:
        """
        Map s (can be scalar or array) to nearest waypoint index
        based on self.waypoints_s (true arc length).
        """
        s_arr = np.asarray(s, dtype=float)
        # wrap to [0, raceline_length)
        s_wrapped = np.mod(s_arr, self.raceline_length)

        s_wp = self.waypoints_s  # (N,)
        idx = np.searchsorted(s_wp, s_wrapped, side="left")
        idx = np.clip(idx, 1, len(s_wp) - 1)

        left = s_wrapped - s_wp[idx - 1]
        # right = s_wp[idx] - s_wp[idx]
        right = s_wp[idx] - s_wrapped

        choose_left = left <= right
        idx_final = np.where(choose_left, idx - 1, idx)

        return idx_final

    def get_frenet(self, x, y, s=None) -> np.array:
        # Compute Frenet coordinates for a given (x, y) point
        if s is None:
            s = self.get_approx_s(x, y)
            s, d = self.get_frenet_coord(x, y, s)
        else:
            s, d = self.get_frenet_coord(x, y, s)

        return np.array([s, d])

    def get_approx_s(self, x, y) -> float:
        """
        Finds the s-coordinate of the given point by finding the nearest waypoint.
        """
        lenx = len(x)
        dist_x = x - np.tile(self.waypoints_x, (lenx, 1)).T
        dist_y = y - np.tile(self.waypoints_y, (lenx, 1)).T

        idx = np.argmin(np.linalg.norm([dist_x.T, dist_y.T], axis=0), axis=1)
        s_approx = self.waypoints_s[idx]                            
        return s_approx

    def get_frenet_velocities(self, vx: float, vy: float, theta: float, s: float) -> np.array:
        """
        Returns the Frenet velocities for the given Cartesian velocities.
        """
        if self.waypoints_psi is None:
            raise ValueError("FRENET CONVERTER: waypoints_psi is None, provide psi to use frenet velocities when initializing the converter.")

        s_idx = int(self._s_to_index(np.array([s]))[0])              

        delta_psi = theta - self.waypoints_psi[s_idx]
        s_dot = vx * np.cos(delta_psi) - vy * np.sin(delta_psi)
        d_dot = vx * np.sin(delta_psi) + vy * np.cos(delta_psi)

        return np.array([s_dot, d_dot])

    def get_frenet_coord(self, x, y, s, eps_m=0.01) -> float:
        """
        Finds the s-coordinate of the given point, considering the perpendicular
        projection of the point on the track.
        """
        _, projection, d = self.check_perpendicular(x, y, s, eps_m)
        for i in range(self.iter_max):
            cand_s = (s + projection) % self.raceline_length
            _, cand_projection, cand_d = self.check_perpendicular(x, y, cand_s, eps_m)

            if self.waypoints_distance_m is not None and self.waypoints_distance_m > 0.0:
                max_step = self.waypoints_distance_m / (2 * self.iter_max)
            else:
                max_step = eps_m  # fallback

            cand_projection = np.clip(cand_projection, -max_step, max_step) 

            updated_idxs = np.abs(cand_projection) <= np.abs(projection)
            d[updated_idxs] = cand_d[updated_idxs]
            s[updated_idxs] = cand_s[updated_idxs]
            projection[updated_idxs] = cand_projection[updated_idxs]

        return s, d

    def check_perpendicular(self, x, y, s, eps_m=0.01) -> Union[bool, float]:
        # obtain unit vector parallel to the track
        dx_ds, dy_ds = self.get_derivative(s)
        tangent = np.array([dx_ds, dy_ds])
        if np.any(np.isnan(s)):
            raise ValueError("BUB FRENET CONVERTER: S is nan")
        tangent /= np.linalg.norm(tangent, axis=0)

        # obtain vector from the track to the point
        x_vec = x - self.spline_x(s)
        y_vec = y - self.spline_y(s)
        point_to_track = np.array([x_vec, y_vec])

        # projection of point_to_track on tangent
        proj = np.einsum('ij,ij->j', tangent, point_to_track)
        perps = np.array([-tangent[1, :], tangent[0, :]])
        d = np.einsum('ij,ij->j', perps, point_to_track)

        check_perpendicular = None
        return check_perpendicular, proj, d

    def get_derivative(self, s) -> np.array:
        """
        Returns the derivative of the point corresponding to s on the chosen line. 
        """
        s = s % self.raceline_length
        der = [self.spline_x(s, 1), self.spline_y(s, 1)]
        return der

    def get_cartesian(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)

        return np.array([x, y])
