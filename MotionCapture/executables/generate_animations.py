import os.path
from scripts.camera import System
import numpy as np

# Define the path to the mocap data
sys = System(calib_path='None')
pred_path = r"../../data/MotionCapture"

# Generate the video for the submissive action
pred_file = "SubmissiveMocap.npy"
title_prefix = "Submissive Action: "

# Generate the video for the walking action
pred_file = "WalkMocap.npy"
title_prefix = "Walk Action: "

# Defining the markers and skeleton
sys.nb_markers = 42
sys.marker_pairs = (
(1, 2), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6), (6, 9), (9, 10), (9, 11), (9, 12), (12, 13), (13, 14),
(13, 15), (14, 15), (14, 16), (15, 17), (16, 17), (17, 18), (17, 19), (16, 19), (19, 20),
(12, 21), (21, 22),
(21, 23), (22, 23), (22, 24), (23, 25), (24, 25), (25, 26), (25, 27), (24, 27), (27, 28),
(12, 29),
(29, 40), (40, 30),
(40, 35),
(30, 31),
(31, 32),
(32, 33),
(32, 34),
(35, 36),
(36, 37),
(37, 38),
(37, 39),
(40, 41),
(41, 42))

sys.nb_individuals = 1
sys.vid_path = pred_path

# Load the poses
sys.poses = np.load(os.path.join(pred_path, pred_file), allow_pickle=True)[()]

# Set limits to the animation
limits = [[-350, 200], [-150, 450], [-50, 400]]

# Generate the animation in the same folder
sys.show_3d_poses_as_video(show_skeleton=True, lims=limits, title_prefix=title_prefix)
