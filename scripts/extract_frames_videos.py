import os
from os.path import join
import cv2
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt

class VideoExtractor(QMainWindow):

    def __init__(self, path, video_extension='.avi'):

        super().__init__()

        # Save video path
        self.video_path = path

        self.session, self.action = self.retrieve_session_info()

        # Get all video files
        self.video_files = [x for x in os.listdir(path) if video_extension in x]

        # Extract camera names
        self.camera_names = []
        self.get_camera_names_from_file_names()
        self.nb_cameras = len(self.camera_names)

        # Sort the names and video_files
        self.video_files = [self.video_files[i] for i in np.argsort(self.camera_names)]
        self.camera_names = np.sort(self.camera_names)

        # Placeholders for pyqtgraph

        # Create a GraphicsLayoutWidget
        self.graphWidget = pg.GraphicsLayoutWidget()
        # Set the GraphicsLayoutWidget as the central widget of the MainWindow
        self.setCentralWidget(self.graphWidget)

        # Create a plot area (which supports titles) and add it to the layout
        self.plotItem = self.graphWidget.addPlot()
        self.plotItem.invertY(True)

        # Create an ImageItem (but don't add it to the layout yet)
        self.imageItem = pg.ImageItem(axisOrder = 'row-major')

        # Add the Imageitem
        self.plotItem.addItem(self.imageItem)

        # Placeholder for current frame in video
        self.frame_number = 0
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

        # Image placeholder
        self.frame = 0

        # Amount of frames to move with arrow keys
        self.move_frames = 10


        self.cur_camera_index = 0


        # Grab all videos to get min_nb_frames
        self.nb_frames_cam = {}
        self.min_frames = 0

        self.retrieve_min_frames_videos()

        # Grab the first frame
        self.grab_video()
        self.init_frame()

    def retrieve_min_frames_videos(self):



        frames_dict = {}
        frames_list = []

        # Open video files and get common number of images
        for f in self.video_files:
            vid = cv2.VideoCapture(os.path.join(self.video_path, f))
            frames_dict[f] = vid.get(cv2.CAP_PROP_FRAME_COUNT)
            frames_list.append(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            vid.release()

        frames_list = np.array(frames_list)
        min_frames = int(frames_list.min())

        self.nb_frames_cam = frames_dict
        self.min_frames = min_frames
        self.frame_count = min_frames

        print(frames_dict)
        print(min_frames)


    def retrieve_session_info(self):

        pos_ = self.video_path.find("Session")

        reduced_path = self.video_path[pos_:]

        pos_2 = reduced_path.find("/")

        session_string = self.video_path[pos_:pos_+pos_2]

        print(session_string)

        pos_action_start = self.video_path.rfind("/")
        action_string = self.video_path[pos_action_start + 1:]

        return session_string, action_string


    def show_image(self):

        # Set the image data
        self.imageItem.setImage(self.frame)

        title = f"{self.session}, Action: {self.action}, Camera: {self.camera_names[self.cur_camera_index]}, Frame: {self.frame_number}"
        self.plotItem.setTitle(title)


    def update_frame(self):

        self.im = self.ax.imshow(self.frame)
        self.ax.set_title(f"{self.session}, Action: {self.action}, Camera: {self.camera_names[self.cur_camera_index]}, Frame: {self.frame_number}")
        self.figure.canvas.draw()


    def init_frame(self):

        self.show_image()


    def get_camera_names_from_file_names(self):

        for v_file in self.video_files:

            pos_ = v_file.rfind("_")

            self.camera_names.append(int(v_file[pos_+1:-4]))

    def grab_video(self):

        cap = cv2.VideoCapture(join(self.video_path, self.video_files[self.cur_camera_index]))

        #self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set the current frame
        # cap.set(cv2.CAP_PROP_FRAME_COUNT, self.frame_number)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)

        ret, frame = cap.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()


    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Right:

            self.frame_number += self.move_frames

            if self.frame_number > self.frame_count:
                self.frame_number = self.frame_count

            print(self.frame_number)

            self.grab_video()
            self.show_image()

        elif event.key() == Qt.Key_Left:

            self.frame_number -= self.move_frames

            if self.frame_number < 0:
                self.frame_number = 0

            print(self.frame_number)

            self.grab_video()
            self.show_image()

        elif event.key() == Qt.Key_Up:
            self.cur_camera_index += 1

            if self.cur_camera_index == self.nb_cameras:
                self.cur_camera_index = self.nb_cameras - 1

            self.grab_video()
            self.show_image()
        elif event.key() == Qt.Key_Down:
            self.cur_camera_index -= 1

            if self.cur_camera_index < 0:
                self.cur_camera_index = 0

            self.grab_video()
            self.show_image()
        elif event.key() == Qt.Key_R:
            self.save_images()
        elif event.key() == Qt.Key_E:
            self.close()
        elif event.key() == Qt.Key_Plus:
            self.move_frames += 1
            if self.move_frames > self.frame_count // 2:
                self.move_frames = self.frame_count // 2
            print("Move Frames: ", self.move_frames)
        elif event.key() == Qt.Key_Minus:
            self.move_frames -= 1
            if self.move_frames < 1:
                self.move_frames = 1
            print("Move Frames: ", self.move_frames)


    def save_images(self):

        labeled_img_path = join(self.video_path, "labeled_images")
        if not os.path.exists(labeled_img_path):
            os.mkdir(labeled_img_path)

        for camera_name in self.camera_names:
            camera_folder_path = join(labeled_img_path, f"cam{camera_name}")
            if not os.path.exists(camera_folder_path):
                os.mkdir(camera_folder_path)

        # Save the respective images
        for camera_index, camera_name in enumerate(self.camera_names):


            cap = cv2.VideoCapture(join(self.video_path, self.video_files[camera_index]))

            # Get the total number of frames
            nb_frames_vid = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            add = 0

            if nb_frames_vid > self.min_frames:
                add = 1

            frame_index = add + self.frame_number

            print(f"Cam{camera_name}: {frame_index}")

            # Set the current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            # Retrieve it
            ret, frame = cap.read()

            # Save the image to the according path
            cam_path = join(labeled_img_path, f"cam{camera_name}")
            cv2.imwrite(join(cam_path, f"{self.frame_number}.png"), frame)

            cap.release()



