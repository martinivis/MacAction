# MacAction
Repository containing code for the paper: 
MacAction: Realistic 3D macaque body animation based on multi-camera markerless motion capture
https://www.biorxiv.org/content/10.1101/2024.01.29.577734v2


# Structure
To run code examples, please first download the data at: ...
and put the data in a root-folder **data** in this project.

```
MacAction
│   README.md
│   AnimLabel
|   MotionCapture    
│   UncannyValley
└───data
│   └───AnimLabel
│   └───MotionCapture
│   └───UncannyValley
```

## AnimLabel Software
Three scripts that can be executed.

1. _run_frame_extraction.py_: Extract frames from a video so you can label them
2. _run_labeling.py_: Run the multi-animal multi-view labeling pipeline with reprojection and epipolar lines
3. _gamma_and_compression.py_: Gamma correct, resize, and compress videos.  

FrameExtractor (1.) Usage: Extract frames of an action that you want to track
- &larr; and &rarr; : Navigate frames within a video
- &uarr; and &darr; : Navigate the cameras within the scene for a specific frame
- r : Retrieve the current frame across cameras and save it to disc
- +: Increase the number of frames to move in time by 1
- -: Decrease the number of frames to move in time by 1
- e: Exit the application

AnimLabel (2.) Usage: Label an action that you extracted frames of, Start with the first frame and click the markers from multiple views of the same location, then move to the next marker by pressing n, continue until done. re-projections appear after two clicked points and epipolar lines can help (toggle t) for labeling. Changing the individual can be achieved by pressing i, and the keyframe to label with arrow keys. Afterwards, save the data by pressing d.

- &larr; and &rarr; : Navigate frames within a video
- Mouse click: Click a point and make it marked
- Mouse wheel: Zoom in and out
- Mouse wheel pushed: Pan across the image
- c: After re-arranging the camera windows for labeling, press c to save the configuration for continuing labeling in the same configuration the next time
- m: Toggle the current marker to be labeled or not labeled
- After one clicked point: epipolar lines are visible if (on, toogle with t)
- After two clicked points, the points are reprojected into other views
- n: Next marker to label, only if mean pixel error is below 5
- 0-9: Mapping to the markers by number
- a: Change to a higher marker cycle
- i: Change individual to label
- d: Save the labels
- e: Exit the application and backup labels with timestamp

## MotionCapture
_generate_animations.py_: Creates matplotlib animations for visualization of the tracked 3d trajectories of the two actions used for the uncanny valley analysis
## UncannyValley
_EyeFixationAnalysis.ipynb_: A jupyter notebook that loads all data from the experiment, processes or pre-loads the processed data, and generates the plots of the paper


# Citing
If you find the code useful for your research, please consider citing the following paper:

```bibtex
@article {Martini2024.01.29.577734,
	author = {Lucas M. Martini and Anna Bogn{\'a}r and Rufin Vogels and Martin A. Giese},
	title = {MacAction: Realistic 3D macaque body animation based on multi-camera markerless motion capture},
	year = {2024},
	doi = {10.1101/2024.01.29.577734},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/04/08/2024.01.29.577734},
	eprint = {https://www.biorxiv.org/content/early/2024/04/08/2024.01.29.577734.full.pdf},
	journal = {bioRxiv}
}```