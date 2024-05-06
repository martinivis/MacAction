import scripts.utils as utils


# Specify the video path that you want to compress, resize, or change in gamma
vid_path = r"../../data/AnimLabel/"

gamma = 0.5
lossy= True
resize = 0.5

# Adjust
utils.adjust_videos_per_path(vid_path, gamma=gamma, resize=resize, lossy=lossy)