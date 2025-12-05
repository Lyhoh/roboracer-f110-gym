import os
import csv
from pathlib import Path

map_name = "IMS"
this_file = Path(__file__).resolve()
repo_root = this_file.parents[2]
tracks_dir = repo_root / "f1tenth_gym_ros" / "maps" / "f1tenth_racetracks"
center_csv = os.path.join(tracks_dir, map_name, f"{map_name}_centerline.csv")
race_csv = os.path.join(tracks_dir, map_name, f"{map_name}_raceline.csv")

with open(center_csv, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = float(row["s_m"])
        x = float(row["x_m"])
        y = float(row["y_m"])
        dl = float(row["d_left"])
        dr = float(row["d_right"])