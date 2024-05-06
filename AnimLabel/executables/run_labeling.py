import sys
from scripts.label import Labeling


# Calibration and video path
calib_path = r"../../data/AnimLabel/calibration.toml"
vid_path = r"../../data/AnimLabel/"

# Labels and individuals in scene
labels = ["Neck","Nose","BodyCenter","lShoulder","lElbow","lWrist","lHip","rShoulder","rElbow","rWrist","rHip","lEye",
          "lEar","rEye","rEar"]

names_individuals=['Ind1', 'Ind2', 'Ind3', 'Ind4', 'Ind5', 'Ind6']

# Configuration path for canvas
config_path = r"../../data/AnimLabel/canvas_config.yaml"

labels = labels
prev_labels = None
nb_markers = labels.__len__()
cropping = False
copy_labels = False

# Use these to reset some labels you want to relabel
reset_labels = []

# Init Labeling system
label_sys = Labeling(calib_path, vid_path, nb_markers, names_individuals, load_cropped=cropping,
                     copy_labels=copy_labels, label_names=labels, bodyparts_names=None, reduced_bp=None,
                     prev_labels=prev_labels, reset_labels=reset_labels, config_path=config_path)

# Start labeling
label_sys.load_keyframe()
label_sys.main_window.hide()
sys.exit(label_sys.app.exec_())

