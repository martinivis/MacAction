from scripts.extract_frames_videos import VideoExtractor
from PyQt5.QtWidgets import QApplication
import sys


# Video path to extract frames from
vid_path = r"../../data/AnimLabel/"


# Start the app to extract frames
app = QApplication(sys.argv)
vid_extractor = VideoExtractor(vid_path)
vid_extractor.show()
sys.exit(app.exec_())