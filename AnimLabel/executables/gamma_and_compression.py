import sys
import os
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

# Append and use
if project_root not in sys.path:
    sys.path.append(project_root)

import scripts.utils as utils


# Specify the video path that you want to compress, resize, or change in gamma
vid_path = r"../../data/AnimLabel/"

gamma = 0.5
lossy= True
resize = 2

# Adjust
utils.adjust_videos_per_path(vid_path, gamma=gamma, resize=resize, lossy=lossy)