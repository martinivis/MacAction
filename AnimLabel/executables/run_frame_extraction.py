import sys
import os
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

# Append and use
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.extract_frames_videos import VideoExtractor
from PyQt5.QtWidgets import QApplication


# Video path to extract frames from
vid_path = r"../../data/AnimLabel/"


# Start the app to extract frames
app = QApplication(sys.argv)
vid_extractor = VideoExtractor(vid_path)
vid_extractor.show()
sys.exit(app.exec_())