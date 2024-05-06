from scripts.camera import System
import numpy as np
import os
import cv2
import matplotlib.image as mpimg
from os.path import join
from matplotlib.markers import MarkerStyle
from datetime import datetime as dt
import scripts.utils as utils
import copy
import shutil
import csv
import math
from ruamel.yaml import YAML

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, pyqtSignal


class GraphWindow(pg.GraphicsLayoutWidget):

    dotPlacedSignal = pyqtSignal(float, float, str)

    keyframeChangedSignal = pyqtSignal(int)

    toggleEpipolarSignal = pyqtSignal()

    saveConfigSignal = pyqtSignal()

    nextMarkerSignal = pyqtSignal()

    changeMarkerSignal = pyqtSignal(int)

    changeMarkerCycleSignal = pyqtSignal()

    changeIndividualSignal = pyqtSignal()

    occlusionChangedSignal = pyqtSignal(str)

    saveLabelsSignal = pyqtSignal()

    exitSignal = pyqtSignal()

    flipIndSignal = pyqtSignal()

    def __init__(self, window_container, title="More specific stuff", camera="0", *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.setWindowTitle(title)


        self.main_window = window_container

        # Create a title of the plot
        self.plot = self.addPlot(title=title)

        self.plot.invertY(True)

        # Empty image
        self.image = None
        self.camera = camera

        # By index, consists of (x, y, occ, dot_object, label_object)
        self.points = []
        self.lines = {}


        ## Legend
        # Initialize the legend
        self.legend = pg.LegendItem(offset=(0, 0))
        self.legend.setParentItem(self.plot.graphicsItem())  # add legend to plot

        # Position the legend at the top-right corner of the plot area
        self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))


        # Show the figure
        self.show()

        # Other settings
        self.marker_colors = [('r'), ('b')]
        self.l_color = 'y'
        self.pen = pg.mkPen(color='y', style=Qt.DashLine)

    def addPoint(self, x, y, occ, label_str):

        occ = int(occ)

        # Change color by occ value
        plotDataItem = self.plot.plot([x], [y], symbol='o', symbolSize=10, symbolBrush=self.marker_colors[occ])

        # Create and add a label for the new marker
        label = pg.TextItem(text=label_str, color=(255, 255, 255))
        self.plot.addItem(label)
        label.setPos(x, y)

        self.points.append((x, y, occ, plotDataItem, label))

    def addLine(self, xs, ys, cam_name):

        plotDataItem = self.plot.plot(xs, ys, pen=self.pen)
        plotDataItem.setVisible(0)

        self.lines[cam_name] = plotDataItem


    def changePoint(self, x, y, occ, index):

        occ = int(occ)

        point = self.points[index][3]
        point.setData([x], [y])

        point.setSymbolBrush(self.marker_colors[occ])

        # likely not here as the change point method is used for clicking as well
        label = self.points[index][4]
        label.setPos(x, y)

        self.points[index] = (x, y, occ, point, label)

    def update_title(self, new_title):
        self.plot.setTitle(new_title)

    def add_image(self, image):

        # Save the image in the class
        self.image = image

        height, width = image.shape[:2]

        # Create an ImageItem
        self.imgItem = pg.ImageItem(image=self.image, axisOrder='row-major')

        # Add the image item to the plot
        self.plot.addItem(self.imgItem)

        # Set the limits
        self.plot.getViewBox().setLimits(xMin=0, xMax=width, yMin=0, yMax=height)

    def set_image(self, image):

        # Set the new Image as image
        self.imgItem.setImage(image)

        height, width = image.shape[:2]

        # Save the image
        self.image = image

        # Set the limits
        self.plot.getViewBox().setLimits(xMin=0, xMax=width, yMin=0, yMax=height)


    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Right:
            self.keyframeChangedSignal.emit(1)
        elif event.key() == Qt.Key_Left:
            self.keyframeChangedSignal.emit(-1)
        elif event.key() == Qt.Key_T:
            self.toggleEpipolarSignal.emit()
        elif event.key() == Qt.Key_C:
            self.saveConfigSignal.emit()
        elif event.key() == Qt.Key_N:
            self.nextMarkerSignal.emit()
        elif event.key() >= Qt.Key_0 and event.key() <= Qt.Key_9:
            self.changeMarkerSignal.emit(int(event.key() - Qt.Key_0))
        elif event.key() == Qt.Key_A:
            self.changeMarkerCycleSignal.emit()
        elif event.key() == Qt.Key_I:
            self.changeIndividualSignal.emit()
        elif event.key() == Qt.Key_M:
            self.occlusionChangedSignal.emit(self.camera)
        elif event.key() == Qt.Key_D:
            self.saveLabelsSignal.emit()
        elif event.key() == Qt.Key_E:
            self.exitSignal.emit()
        elif event.key() == Qt.Key_F:
            self.flipIndSignal.emit()
        else:
            super().keyPressEvent(event)


    def mousePressEvent(self, event):
        # Check for left mouse button press
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            clickedPoint = self.plot.vb.mapSceneToView(pos)

            if self.imgItem.boundingRect().contains(clickedPoint):
                # Emit signal to parent class
                self.dotPlacedSignal.emit(clickedPoint.x(), clickedPoint.y(), self.camera)
        else:
            # Pass other mouse events to the parent class
            super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, LabelSys):
        super().__init__()
        self.graph_windows = {}
        self.labelsys = LabelSys

    def add_graph_window(self, cam):
        title = f"Camera View {cam}"
        graph_window = GraphWindow(window_container=self, title=title, camera=cam)


        # Add the Signal
        graph_window.dotPlacedSignal.connect(self.dotPlaced)
        graph_window.keyframeChangedSignal.connect(self.keyframeChange)

        graph_window.toggleEpipolarSignal.connect(self.toggleEpipolar)
        graph_window.saveConfigSignal.connect(self.saveConfig)
        graph_window.nextMarkerSignal.connect(self.nextMarker)
        graph_window.changeMarkerSignal.connect(self.changeMarker)
        graph_window.changeMarkerCycleSignal.connect(self.changeMarkerCycle)
        graph_window.changeIndividualSignal.connect(self.changeIndividual)
        graph_window.occlusionChangedSignal.connect(self.occlusionChanged)
        graph_window.saveLabelsSignal.connect(self.saveLabels)
        graph_window.exitSignal.connect(self.exit)

        graph_window.flipIndSignal.connect(self.labelsys.flip_markers_of_individuals_per_frame)

        # Add to dictionary
        self.graph_windows[cam] = graph_window

        return graph_window

    def toggleEpipolar(self):

        self.labelsys.epipolar_toggle = not self.labelsys.epipolar_toggle

        if self.labelsys.epipolar_toggle:
            # Show them
            points_labeled, labeled_cams = self.labelsys.get_nb_vis_points()

            # If there are more than 2, lift markers
            if points_labeled > 0:
                if self.labelsys.epipolar_toggle:
                    # Draw lines
                    self.labelsys.draw_epipolar_lines(labeled_cams)
        else:
            self.labelsys.make_lines_invisible()


    def saveConfig(self):

        geometry_windows = {}

        for cam, graph_window in self.graph_windows.items():
            geometry_windows[str(cam)] = {'geometry': graph_window.geometry().getRect()}
                                     #'viewRange': graph_window.plot.viewRange()}

        self.labelsys.canvas_configuration = geometry_windows
        self.labelsys.save_details(update=True)

    def nextMarker(self):

        if self.labelsys.cur_mean_rep_er < 5:
            if self.labelsys.marker_selected + 1 < self.labelsys.nb_markers:
                self.labelsys.marker_selected += 1
            else:
                self.labelsys.marker_selected = 0

            self.labelsys.marker_cycle, _ = divmod(self.labelsys.marker_selected + 1, 10)

            self.labelsys.cur_mean_rep_er = -1
            # Reset rep error per marker in each frame to -1 either way, maybe make this with calculus, but not sure
            for cam_name in self.labelsys.sys_dict.keys():
                self.labelsys.cur_key_frame[cam_name][-1] = -1
            # redraw routine erst nach click callen nicht nach anderer auswahl, bisschen schneller.
            # das vllt auch nach calculations erst für reprojection.
            # Undraw epipolar lines
            self.labelsys.make_lines_invisible()
            self.labelsys.update_title_figures()

        else:


            print(f"You probably missed updating a label in {self.labelsys.cur_frame}. Marker: {self.labelsys.label_names[self.labelsys.marker_selected]}")


    def changeMarker(self, marker_number):

        self.labelsys.update_cur_marker(event_key=marker_number)
        self.labelsys.cur_mean_rep_er = -1
        # Reset rep error per marker in each frame to -1 either way, maybe make this with calculus, but not sure
        for cam_name in self.labelsys.sys_dict.keys():
            self.labelsys.cur_key_frame[cam_name][-1] = -1

        self.labelsys.make_lines_invisible()
        self.labelsys.update_title_figures()

    def changeMarkerCycle(self):

        if self.labelsys.marker_cycle + 1 > self.labelsys.max_cycle:
            self.labelsys.marker_cycle = 0
        else:
            self.labelsys.marker_cycle += 1
        self.labelsys.update_cur_marker()
        self.labelsys.update_title_figures()

    def changeIndividual(self):

        new_individual = self.labelsys.ind_selected + 1

        # Individuals is 0-indexed
        if new_individual < self.labelsys.nb_individuals:
            self.labelsys.ind_selected = new_individual
        else:
            # If new individual index equals nb individuals,
            # map to first individual again
            self.labelsys.ind_selected = 0
        self.labelsys.update_marker_shift()
        self.labelsys.update_title_figures()

    def occlusionChanged(self, camera):

        self.labelsys.occlusion_change(camera)


    def saveLabels(self):
        print("Saving labels")
        self.labelsys.save_labels()

    def exit(self):

        print("Backup before exiting")
        self.labelsys.save_labels(alter_string=f"_{str(dt.now())[:-7].replace(':', '-').replace(' ', '_')}")

        self.labelsys.print_feedback_labeling()


        for graphwindow in self.graph_windows.values():
            graphwindow.close()

        # Close the main window
        self.close()

        # Quit main application
        QApplication.instance().quit()


    def dotPlaced(self, x, y, cam):
        # Called when a dot is placed in any GraphWindow


        # if clicked then add UL to the pixel position
        if self.labelsys.load_cropped:
            [UL, LR] = self.labelsys.init_crop_params[str(self.labelsys.cur_frame)][cam]
            x += UL[0]
            y += UL[1]

        # Change other markers that are not labeled based on reprojection,
        # and print_rep_error on figures where markers were put
        # Save to labels

        self.labelsys.labels[cam][self.labelsys.marker_selected + self.labelsys.marker_shift, :, self.labelsys.cur_frame] = np.array([x, y])
        self.labelsys.occs[cam][self.labelsys.marker_selected + self.labelsys.marker_shift, self.labelsys.cur_frame] = 0

        # Update marker artists as well
        self.labelsys.update_marker_and_text_pos(cam, make_vis=True)

        self.labelsys.check_reprojection()


    def keyframeChange(self, direction):

        if direction > 0:
            # Right

            if self.labelsys.frame_ind + 1 < self.labelsys.frames_labeled_indices.__len__():
                self.labelsys.frame_ind += 1
                self.labelsys.load_keyframe()

        else:
            # Left
            if self.labelsys.frame_ind - 1 >= 0:
                self.labelsys.frame_ind -= 1
                self.labelsys.load_keyframe()




# p is used for the transitioning tool
# n can be used to set to next marker
class Labeling(System):
    def __init__(self, calib_path, vid_path, nb_markers, names_individuals=['Ind1'], load_cropped=False,
                 copy_labels=False, label_names=None, bodyparts_names=None, prev_labels=None, reduced_bp=None,
                 reset_labels=[], image_extension='png', config_path=None):




        super().__init__(calib_path)

        self.load_cropped = load_cropped
        self.vid_path = vid_path

        self.init_crop_params = None
        self.pix_area_inc = 50

        self.fundamentals = {}

        # For labeling, exporting and transfering labels


        self.label_names = label_names
        self.bodyparts_names = bodyparts_names
        self.prev_labels = prev_labels
        self.reduced_bp = reduced_bp

        self.image_extension = image_extension


        # change to read video camera parameters if there are

        # Loads the camera calibration file
        self.check_for_recalibration()

        self.epipolar_samples = 2000#1000


        self.fill_fundamentals()
        self.vis_epipolar_lines = {}
        self.nb_markers = nb_markers

        self.ind_selected = 0
        self.marker_shift = 0

        self.names = names_individuals
        self.nb_individuals = len(names_individuals)


        if config_path is None:
            self.config_path = self.vid_path
        else:
            self.config_path = config_path

        # Canvas configuration
        self.canvas_configuration = None

        # Saves details of the setup or load if there are already
        self.save_details()

        # Only takes effect if there is prev_labels not None.
        self.reset_labels = reset_labels


        # Init labels
        self.init_labels()

        if load_cropped:

            if os.path.exists(join(self.vid_path, "maskrcnn_altered_crops.npy")):
                self.init_crop_params = np.load(join(self.vid_path, "maskrcnn_altered_crops.npy"), allow_pickle=True)[()]
            elif os.path.exists(join(self.vid_path,"maskrcnn_init_crop.npy")):
                self.init_crop_params = np.load(join(self.vid_path,"maskrcnn_init_crop.npy"), allow_pickle=True)[()]
            else:
                ValueError("There is no cropping information available!")

            # Get the bounding box that gets them all
            self.max_bb_crop_area()

        # Initially they will not be shown
        self.epipolar_toggle = False


        # change this, to reading frame names in directory
        #self.load_trk_labels(vid_path)

        self.startup = True
        self.frame_ind = 0
        self.cur_frame = 0

        self.last_key = None
        self.sel_corner = None

        # Corresponds to 1 on keyboard
        self.marker_selected = 0
        self.marker_cycle = 0
        self.max_cycle = 0

        a, b = divmod(self.nb_markers, 10)

        if b == 0:
            # 0-indexed cycles
            self.max_cycle = a - 1
        else:
            self.max_cycle = a


        # Selected individual, which will for now share all the markers of first individual


        self.v_marker = MarkerStyle("+")
        self.o_marker = MarkerStyle(".")

        self.v_size = 14
        self.o_size = 3

        self.l_size = .2
        self.l_m_size = 1

        self.v_color = 'r'
        self.o_color = 'b'
        self.l_color = 'y'

        self.cur_key_frame = {}
        self.cam_indices = {}

        self.cur_mean_rep_er = -1
        self.cur_best_set_size = -1
        self.cur_set_size = -1

        self.copy_labels = copy_labels



        ## QT Plotting related
        self.app = QApplication([vid_path])
        # Create a main_window
        self.main_window = MainWindow(self)


        # Print infos
        print(f"Calib path: {self.calib_path}")
        print(f"Video path: {self.vid_path}")
        print(f"Individual(s) : {self.names}")
        print(f"Frames: {self.frames_labeled_indices}")

    def save_details(self, update = False):



        # Check if the file exists
        label_info_path = join(self.vid_path, 'labeling_info.yaml')

        # Create new YAML object
        ruamel_file = YAML()


        # Write it
        if update:
            self.write_details(ruamel_file, label_info_path)

        if os.path.isfile(label_info_path):
            # File exists, load it
            with open(label_info_path, "r") as f:
                yaml_file = ruamel_file.load(f)
            self.names = yaml_file['names_individuals']
            self.nb_individuals = len(self.names)
            self.canvas_configuration = yaml_file['canvas_configuration']
        else:
            # Write it
            self.write_details(ruamel_file, label_info_path)

        # Override the config
        if self.config_path is not None and os.path.isfile(self.config_path):
            ruamel_config = YAML()
            with open(self.config_path, "r") as f:
                config_file = ruamel_config.load(f)
            self.canvas_configuration = config_file['canvas_configuration']

    def write_details(self, ruamel_file, label_info_path):

        # Create new yaml file for details
        info_dict = {}

        info_dict['names_individuals'] = self.names


        if self.canvas_configuration is not None:


            if self.config_path == self.vid_path:
                info_dict['canvas_configuration'] = self.canvas_configuration

            else:

                config_ruamel = YAML()
                config_dict = {}
                config_dict['canvas_configuration'] = self.canvas_configuration

                # Dump the config file
                with open(self.config_path, 'w') as f:
                    config_ruamel.dump(config_dict, f)

                info_dict['canvas_configuration'] = None
            print(info_dict)
        else:
            info_dict['canvas_configuration'] = None
        with open(label_info_path, 'w') as f:
            ruamel_file.dump(info_dict, f)

    def init_labels(self):

        # Check if labels in path exist, if yes, load them otherwise init and save

        dir_info = os.listdir(self.vid_path)

        # Only checking for one not occ
        if 'labels.npy' in dir_info:

            print("Load Labels")
            self.load_labels()
            self.vid_length = list(self.labels.values())[0].shape[-1]


            if self.prev_labels is not None:

                # Save loaded old labels
                prev_label_pos = copy.deepcopy(self.labels)
                prev_label_occs = copy.deepcopy(self.occs)

                # Init new labels by new marker set which potentially is of different size
                # Video length should not change
                self.init_default_labels(nb_individuals=self.nb_individuals)


                # Make sure to have the same names in the labels

                # Iter over new bodyparts and check if a similiar one exists in previous marker set
                # If so transfer previous labels into labels

                print(f"The following labels will not be transfered: {self.reset_labels}")

                for new_bp_index, new_bp in enumerate(self.label_names):

                    if new_bp not in self.reset_labels:
                        if new_bp in self.prev_labels:
                            old_bp_index = self.prev_labels.index(new_bp)

                            for cam in self.sys_dict.keys():
                                for frame in self.frames_labeled_indices:

                                    for j in range(self.nb_individuals):
                                        self.labels[cam][new_bp_index+(self.nb_markers*j), :, frame] = \
                                            prev_label_pos[cam][old_bp_index+(self.prev_labels.__len__()*j), :, frame]

                                        self.occs[cam][new_bp_index + (self.nb_markers * j), frame] = \
                                            prev_label_occs[cam][old_bp_index + (self.prev_labels.__len__() * j), frame]

        else:
            # Init labels

            # Get frame indices first
            self.get_frames_indices()
            self.get_nb_frames_vid()

            # Init the default labels based on video length
            self.init_default_labels(nb_individuals=self.nb_individuals)


    def print_feedback_labeling(self):

        # Take any cam

        cam = list(self.sys_dict.keys())[0]

        print_flag = False
        for frame in self.frames_labeled_indices:
            for ind in range(self.nb_individuals):
                for m in range(self.nb_markers):
                    xy = self.labels[cam][m + ind * self.nb_markers, :, frame]

                    if np.sum(xy) == 0:
                        print(f"Frame {frame} seems unlabeled for marker {m} and Individual {ind}!")
                        print_flag = True

        if not print_flag:
            print("Labeling of the action seems complete.")

    def flip_markers_of_individuals_per_frame(self):
        """
        Only for 2 individuals
        :return:
        """

        labels_copy = copy.deepcopy(self.labels)
        occs_copy = copy.deepcopy(self.occs)

        for cam in self.sys_dict.keys():
            for marker in range(self.nb_markers):
                for ind in range(self.nb_individuals):
                    self.labels[cam][marker + ind * self.nb_markers, :, self.cur_frame] = \
                        labels_copy[cam][marker + int(1-ind) * self.nb_markers, : , self.cur_frame]
                    self.occs[cam][marker + ind * self.nb_markers, self.cur_frame] = \
                        occs_copy[cam][marker + int(1-ind) * self.nb_markers, self.cur_frame]


        print("Labels got flipped over individuals for the current frame")

    def fill_fundamentals(self):
        """
        Create all fundamentals for each right camera, with transformations to each left camera
        """

        # Iter cams to draw the line/s
        for cam_name_right, cam_right in self.sys_dict.items():
            fund_dict = {}
            # Iter possible cameras with points in them
            for cam_name_left, cam_left in self.sys_dict.items():
                if cam_name_left != cam_name_right:
                    R1, t1, K1 = [cam_left.R, cam_left.t, cam_left.K]
                    R2, t2, K2 = [cam_right.R, cam_right.t, cam_right.K]

                    #print(t1.shape)
                    #print(R1.shape)
                    fund_dict[cam_name_left] = utils.fundamental_From_RtK(R1, t1, K1, R2, t2, K2)
                else:
                    fund_dict[cam_name_left] = None

            self.fundamentals[cam_name_right] = fund_dict

    def update_marker_shift(self):

        self.marker_shift = self.ind_selected * self.nb_markers

    def flip_occ_all(self):
        # Flip all occlusions.
        for key in self.occs.keys():
            for m in range(self.nb_markers*self.nb_individuals):
                for frame in range(self.vid_length):
                    self.occs[key][m, frame] = not self.occs[key][m, frame]


    def calculate_new_markerselection(self, k):
        """
        k is equivalent to keyboard press
        """

        if k == 0:
            l = 9
        else:
            l = k - 1


        shifted_l = l + 10 * self.marker_cycle

        if shifted_l < self.nb_markers:
            self.marker_selected = shifted_l
        else:
            self.marker_selected = l
            self.marker_cycle = 0


    def update_cur_marker(self, event_key=None):


        # Key press event
        if event_key is not None:
            self.calculate_new_markerselection(int(event_key))
        else:
            # Updated cycle
            k = (self.marker_selected + 1) % 10
            self.calculate_new_markerselection(k)

    def update_marker_and_text_pos(self, cam, make_vis=False):

        x, y = self.labels[cam][self.marker_selected+self.marker_shift,
               :, self.cur_frame][:]

        # Change visible position of projection to cropped area.
        if self.load_cropped:
            [UL, LR] = self.init_crop_params[str(self.cur_frame)][cam]
            x -= UL[0]
            y -= UL[1]

        marker_index = self.marker_selected+self.marker_shift

        graph_window = self.main_window.graph_windows[cam]

        if make_vis:
            graph_window.changePoint(x, y, 0, marker_index)
        else:
            graph_window.changePoint(x, y, 1, marker_index)


    def get_frames_indices(self):
        """
        Get frame indices from images in subfolder
        """

        # Go into first folder of sys_dict, probably cam A
        first_img_folder = join(self.vid_path, join("labeled_images", join(f"cam{list(self.sys_dict.keys())[0]}")))
        # List all files
        dir_info = os.listdir(first_img_folder)

        frame_indices = []
        for file in dir_info:

            if not "rc" in file[:-4]:
                frame = int(file[:-4])
                frame_indices.append(frame)

        # Sort the object
        frame_indices.sort()
        # Assign
        self.frames_labeled_indices = frame_indices


    def check_for_recalibration(self):


        if os.path.isfile(join(self.vid_path, "recalibration.mat")):

            print("Loading RE-calibrated system file")
            self.read_cam_params(alter_path=self.vid_path, extension="recalibration")

        else:
            print("Loading calibration system file")
            self.read_cam_params()

    def get_nb_frames_vid(self):

        dir_info = os.listdir(self.vid_path)

        for file in dir_info:
            # Just pick one
            if ('.avi' in file) or ('.mp4' in file):
                cap = cv2.VideoCapture(join(self.vid_path, file))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.vid_length = length
                cap.release()
                break


    def load_keyframe(self):


        if self.label_names is None:
            raise ValueError("Please define label_names for labeling!")


        frame = self.frames_labeled_indices[self.frame_ind]
        self.cur_frame = frame


        if self.startup:
            frames = {}

            for index, cam in enumerate(self.sys_dict.keys()):
                img_path = join(self.vid_path, join("labeled_images", join(f"cam{cam}", f"{frame}.{self.image_extension}")))
                img = mpimg.imread(img_path)

                height, width = self.sys_dict[cam].im_size

                if self.load_cropped:
                    [UL, LR] = self.init_crop_params[str(frame)][cam]
                    # Crop the image
                    img = img[UL[1]:LR[1], UL[0]:LR[0], :]
                    # Change height and width again
                    height, width = img.shape[:2]


                ## PyQTGRAPH

                # Put titles as well
                m_string = f"[{1 + self.marker_cycle * 10}, {(self.marker_cycle + 1) * 10}]"
                er = -1
                view_title = (
                    f"Cam: {cam}, Frame: {self.cur_frame}, {self.names[self.ind_selected]}, Marker: {self.marker_selected + 1} ({m_string}, {self.label_names[self.marker_selected]}), Error: ({er:.2f}, "
                    f"{self.cur_mean_rep_er:.2f}, {self.cur_best_set_size}, {self.cur_set_size})")

                # Plot the image
                graph_window = self.main_window.add_graph_window(cam)

                # Set canvas information
                if self.canvas_configuration is not None:
                    graph_window.setGeometry(*self.canvas_configuration[cam]['geometry'])

                graph_window.add_image(img)
                graph_window.update_title(view_title)

                # Plot the markers
                # All markers of first individual, then second, ...
                for j in range(self.nb_individuals):
                    for i in range(self.nb_markers):
                        x, y = self.labels[cam][i + (self.nb_markers * j), :, frame]
                        occ = self.occs[cam][i + (self.nb_markers * j), frame]

                        if self.load_cropped:
                            # Add cropping
                            x -= UL[0]
                            y -= UL[1]


                        label_str = f"{i + 1}"
                        graph_window.addPoint(x, y, occ, label_str)



                ## Create all the epipolar lines and set their visibility to 0, (Init)

                line_x = np.arange(0, width, 1)

                vis_ep_lines_cam = {}
                # Be careful not to shadow
                for cam_name in self.sys_dict.keys():

                    graph_window.addLine(line_x, line_x, cam_name)

                    vis_ep_lines_cam[cam_name] = 0

                self.vis_epipolar_lines[cam] = vis_ep_lines_cam
                self.cam_indices[index] = cam

                frames[cam] = [er]






            self.startup = False

            # Creating a show call for pyqtgraph
            self.main_window.show()

            self.cur_key_frame = frames



        else:

            for index, cam in enumerate(self.sys_dict.keys()):
                img_path = join(self.vid_path, join("labeled_images", join(f"cam{cam}", f"{frame}.png")))
                img = mpimg.imread(img_path)

                # go into ax and change image
                #im, fig, ax = [self.cur_key_frame[cam][0], self.cur_key_frame[cam][1], self.cur_key_frame[cam][2]]

                if self.load_cropped:
                    [UL, LR] = self.init_crop_params[str(frame)][cam]

                    # Crop the image
                    img = img[UL[1]:LR[1], UL[0]:LR[0], :]
                    # Change height and width again
                    height, width = img.shape[:2]

                #im.remove()

                # Set
                graph_window = self.main_window.graph_windows[cam]
                graph_window.set_image(img)


                # instead of new scatters, change the data
                for i in range(self.nb_markers*self.nb_individuals):

                    self.check_frame_labeled_per_cam(cam_name=cam)

                    xy = self.labels[cam][i, :, frame]
                    occ = self.occs[cam][i, frame]

                    # Otherwise label position gets shifted
                    xy_disp = copy.deepcopy(xy)

                    # Load points and display them at their new cropped position
                    # but because the crops are not the same the positions will change as well drastically
                    if self.load_cropped:
                        xy_disp[0] -= UL[0]
                        xy_disp[1] -= UL[1]


                    graph_window.changePoint(xy_disp[0], xy_disp[1], occ, i)


                # Undraw lines and set them invisible
                self.set_invisible_flags_epi_lines()
                self.make_lines_invisible()

                # Set errors to default
                self.cur_mean_rep_er = -1
                # Reset rep error per marker in each frame to -1 either way, maybe make this with calculus, but not sure
                for cam_name in self.sys_dict.keys():
                    self.cur_key_frame[cam_name][-1] = -1

            self.update_title_figures()


    def check_frame_labeled_per_cam(self, cam_name):
        """
        Copies labels if the next frame label has (0,0) as a label
        :param cam_name:
        :return:
        """
        # Only forward possible
        #if int(self.occs[cam_name][:, self.cur_frame].sum()) == self.nb_markers*self.nb_individuals:
        if self.frame_ind - 1 >= 0 and self.copy_labels:
            prev_frame = self.frames_labeled_indices[self.frame_ind-1]

            for marker_idx in range(self.nb_markers * self.nb_individuals):

                if int(self.labels[cam_name][marker_idx, :, self.cur_frame].sum()) == 0:
                    #print(marker_idx)
                    self.labels[cam_name][marker_idx, :, self.cur_frame] = self.labels[cam_name][marker_idx, :, prev_frame]
                    self.occs[cam_name][marker_idx, self.cur_frame] = self.occs[cam_name][marker_idx, prev_frame]

    def update_title_figures(self):

        for cam in self.sys_dict.keys():



            graph_window = self.main_window.graph_windows[cam]

            er = self.cur_key_frame[cam][-1]

            m_string = f"[{1+self.marker_cycle*10}, {(self.marker_cycle+1)*10}]"

            new_title = (f"Cam: {cam}, Frame: {self.cur_frame}, {self.names[self.ind_selected]}, Marker: {self.marker_selected + 1} ({m_string}, {self.label_names[self.marker_selected]}), Error: ({er:.2f}, "
                f"{self.cur_mean_rep_er:.2f}, {self.cur_best_set_size}, {self.cur_set_size})")

            graph_window.update_title(new_title)

    def draw_epipolar_lines(self, labeled_cams):

        # Make all invinsible
        self.make_lines_invisible()
        self.set_invisible_flags_epi_lines()

        for labeled_cam in labeled_cams:

            # Get point of labeled cam
            xy = self.labels[labeled_cam][self.marker_selected+self.marker_shift, :, self.cur_frame]

            # Undistort the labeled point
            K1 = self.sys_dict[labeled_cam].K
            d1 = self.sys_dict[labeled_cam].d
            und_xy = utils.undist_points(xy.reshape(1, 1, 2), K1, d1)

            for cam_name in self.sys_dict.keys():

                if labeled_cam == cam_name:
                    pass
                else:
                    # Calculate line parameters
                    line_params = utils.line_params(self.fundamentals[cam_name][labeled_cam], und_xy)

                    # Iterate over all cameras and update the lines
                    line = self.main_window.graph_windows[cam_name].lines[labeled_cam]

                    # Get line x data, negative as well to get more points in image?

                    range_epi_images = 1 # 5
                    line_x = np.linspace(-range_epi_images*self.sys_dict[cam_name].im_size[0],
                                         range_epi_images*self.sys_dict[cam_name].im_size[0], self.epipolar_samples)
                    #line_x = np.arange(-5*self.sys_dict[cam_name].im_size[0], 5*self.sys_dict[cam_name].im_size[0], 10)

                    # Calculate y values
                    line_y = -(line_params[0]/ line_params[1]) * line_x - line_params[2]/ line_params[1]

                    # Distort the sampled points
                    K2 = self.sys_dict[cam_name].K
                    d2 = self.sys_dict[cam_name].d

                    line_dist = np.zeros(shape=(line_x.shape[0], 2))

                    for i in range(line_x.shape[0]):
                        x_d, y_d = utils.distort_point(line_x[i], line_y[i], K2, d2)
                        line_dist[i, 0] = x_d
                        line_dist[i, 1] = y_d


                    #line.set_offsets(line_dist)

                    # Move the projected line to the cropped area region.
                    if self.load_cropped:
                        [UL, LR] = self.init_crop_params[str(self.cur_frame)][cam_name]
                        line_dist[:, 0] -= UL[0]
                        line_dist[:, 1] -= UL[1]

                    line_dist = line_dist.T

                    # Set data and make it visible
                    line.setData(line_dist[0, :], line_dist[1, :])
                    line.setVisible(1)

                    # Push Updates to that line
                    line.update()

                    # Set visibility flag
                    self.vis_epipolar_lines[cam_name][labeled_cam] = 1


    def make_lines_invisible(self):
        for cam in self.sys_dict.keys():
            for cam2 in self.sys_dict.keys():
                self.main_window.graph_windows[cam].lines[cam2].setVisible(0)

    def set_invisible_flags_epi_lines(self):
        for cam in self.sys_dict.keys():
            for cam2 in self.sys_dict.keys():
                self.vis_epipolar_lines[cam][cam2] = 0

    def max_bb_crop_area(self):

        for cam in self.sys_dict.keys():

            min_y = 10000
            max_y = 0
            min_x = 10000
            max_x = 0

            for frame in self.frames_labeled_indices:
                [UL, LR] = self.init_crop_params[str(frame)][cam]

                if UL[0] < min_x:
                    min_x = UL[0]
                if UL[1] < min_y:
                    min_y = UL[1]
                if LR[0] > max_x:
                    max_x = LR[0]
                if LR[1] > max_y:
                    max_y = LR[1]

            for frame in self.frames_labeled_indices:
                self.init_crop_params[str(frame)][cam] = [np.array([min_x, min_y]),
                                                          np.array([max_x, max_y])]



    def check_reprojection(self):
        # Check how many were already labeled, change to array?
        points_labeled, labeled_cams = self.get_nb_vis_points()

        # If there are more than 2, lift markers
        if points_labeled > 0:

            if self.epipolar_toggle:

                # Draw lines
                self.draw_epipolar_lines(labeled_cams)

            if points_labeled >= 2:
                vis_m_dict = {}
                for cam_name in self.sys_dict.keys():
                    if self.occs[cam_name][self.marker_selected+self.marker_shift, self.cur_frame] == 0:
                        vis_m_dict[cam_name] = self.labels[cam_name][self.marker_selected+self.marker_shift, :, self.cur_frame]
                r, rep_error, best_set, best_index, triang_dic = \
                    self.lift_marker(vis_labels=vis_m_dict, undist=True, ransac=True, ransac_minimal=False)

                # Error dictionary
                errors = triang_dic[best_index][2]

                # Set mean rep error
                self.cur_mean_rep_er = rep_error

                # Set best and whole set size
                self.cur_set_size = errors.__len__()
                self.cur_best_set_size = best_set.__len__()

                for cam_name in self.sys_dict.keys():
                    if cam_name in errors:
                        self.cur_key_frame[cam_name][-1] = errors[cam_name]
                    else:
                        # reproject single
                        m_rep = self.reproject_single(frame=0, name=cam_name, p_3d=r)

                        self.labels[cam_name][self.marker_selected+self.marker_shift, :, self.cur_frame] = m_rep

                        if cam_name in labeled_cams:
                            self.update_marker_and_text_pos(cam_name, make_vis=True)
                        else:
                            self.update_marker_and_text_pos(cam_name)


                        self.cur_key_frame[cam_name][-1] = -1

            # Redraw all, because of projected points
            self.update_title_figures()

        else:

            # Make all lines invisible, f.ex. when changing marker type
            self.make_lines_invisible()
            self.set_invisible_flags_epi_lines()
            self.update_title_figures()


    def occlusion_change(self, cam):

        # Current visibility status
        occ_vis_status = self.occs[cam][self.marker_selected + self.marker_shift, self.cur_frame]

        nb_markers_vis = self.get_nb_vis_points()[0]
        # New status
        new_occ_vis_status = not occ_vis_status
        # Flip marker
        self.occs[cam][self.marker_selected + self.marker_shift, self.cur_frame] = new_occ_vis_status

        # It's inverted make_vis and occlusion
        self.update_marker_and_text_pos(cam, make_vis=occ_vis_status)

        # If you change from 2 to 1
        if (nb_markers_vis == 2) and (new_occ_vis_status == 1):
            # If you change back from lifting then print not defined errors
            self.cur_mean_rep_er = -1
            self.cur_set_size = -1
            self.cur_best_set_size = -1
            for cam_name in self.sys_dict.keys():
                self.cur_key_frame[cam_name][-1] = -1

        self.check_reprojection()

    def get_nb_vis_points(self):
        points_labeled = 0
        labeled_cams = []
        for cam_name in self.cam_indices.values():
            if self.occs[cam_name][self.marker_selected+self.marker_shift, self.cur_frame] == 0:
                points_labeled += 1
                labeled_cams.append(cam_name)
        return points_labeled, labeled_cams

    def write_video_labels_to_DLC(self, writer, base_path, DEST, vid_index, all_names_ordered, reduce=False, crop=False, dev_mm=75,
                                  resize=1, names_list=[], convert_to_single_animal=True, image_ext = 'png', reduced_cam_list=None
                                  ):
        """
        Resize only works for non-cropping and if resize is larger than 1
        :param writer:
        :param base_path:
        :param DEST:
        :param vid_index:
        :param reduce:
        :param crop:
        :param dev_mm:
        :param resize:
        :return:
        """

        if resize < 1:
            raise ValueError("Exporting without reducing image size is not supported.")

        # Iter frames labeled:

        crop_params_frames = {}

        for frame in self.frames_labeled_indices:
            crop_params = {}
            for cam_name, camera in self.sys_dict.items():

                cam_name = str(cam_name)

                if reduced_cam_list is not None:
                    # Do only selected cams
                    if cam_name not in reduced_cam_list:
                        continue

                if not crop:

                    if convert_to_single_animal:
                        raise ValueError("Converting multiple individual video sequences "
                                         "to single ones without cropping is unsupported")

                    # Give a unique image name, add resize
                    new_img_name = f'{vid_index}_cam_{cam_name}_{frame}_{resize}.png'

                    # First try png
                    coords = [base_path + new_img_name]

                    # Copy image file
                    #img_path = join(self.vid_path, rf'labeled_images\cam{cam_name}\{frame}.png')
                    dest_img_path = join(DEST, new_img_name)
                    #shutil.copy(img_path, dest_img_path)

                    frame_labels = self.labels[cam_name][:, :, frame]

                    img = self.load_labeled_image(frame, cam_name, extension=image_ext)

                    if resize != 1:
                        # Inter-area is preferred for down-sampling, for up-sampling linear or cubic
                        img = cv2.resize(img, (img.shape[1]//resize, img.shape[0]//resize),
                                         interpolation=cv2.INTER_AREA)

                        # Change label positions by same scale factor
                        frame_labels = frame_labels/resize

                    cv2.imwrite(dest_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


                    self.write_frame_markers_row(writer=writer, coords=coords, labels=frame_labels,
                                                 image_shape=img.shape, all_names_ordered=all_names_ordered, reduce=reduce, vid_index=vid_index,
                                                 names_list=names_list, convert_to_single_animal=convert_to_single_animal
                                                 )

                else:

                    if resize != 1:
                        raise ValueError("Cropping and resizing at the same time is not supported yet.")

                    crop_params_inds = {}

                    # Iter names
                    for ind_index, individual in enumerate(names_list[vid_index]):


                        new_img_name = f'{vid_index}_cam_{cam_name}_{frame}_{ind_index}_cropped.png'

                        # First try png
                        coords = [base_path + new_img_name]


                        frame_labels = self.labels[cam_name][ind_index*self.nb_markers:(ind_index+1)*self.nb_markers, :, frame]
                        img = self.load_labeled_image(frame, cam_name, image_ext=image_ext)

                        # Get the 3d cropping information
                        [UL, LR] = self.crop_3d(frame, cam_name, dev_mm=dev_mm, ind_index=ind_index)

                        # Check for image area of cropping, and if suggested corners are not within image area
                        # Force the limits
                        cropped_img, [UL, LR] = utils.possible_cropping(img, UL, LR)

                        # Save params in dictionary
                        crop_params_inds[individual] = [UL, LR]

                        # Shift the labels
                        shifted_labels = copy.deepcopy(frame_labels)
                        shifted_labels[:, 0] -= UL[0]
                        shifted_labels[:, 1] -= UL[1]

                        # Write cropped image
                        dest_img_path = join(DEST, new_img_name)
                        #plt.imsave(dest_img_path, cropped_img)
                        # Write image with opencv, convert color from rgb to bgr
                        cv2.imwrite(dest_img_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

                        self.write_frame_markers_row(writer=writer, coords=coords, labels=shifted_labels,
                                                 image_shape=cropped_img.shape, all_names_ordered=all_names_ordered, reduce=reduce, vid_index=vid_index,
                                                 names_list=names_list, convert_to_single_animal=convert_to_single_animal)

                    crop_params[cam_name] = crop_params_inds

            crop_params_frames[frame] = crop_params

        if crop:
            # Save to vid path, and aggregate afterwards
            #np.save(join(DEST, "cropping_info_DLC_export.npy"), crop_params_frames)
            np.save(join(self.vid_path, "cropping_info_DLC_export.npy"), crop_params_frames)

            return crop_params_frames

    def crop_export_MaskRCNN(self, DEST, VID_PATHS, dev_mm=75):
        """
        Exports all given VID_PATHS with 3d-cropping to a given location, where a dict is saved as npy containing each
        images cropping parameters
        :return:
        """

        crop_vid_params = {}

        for vid_index, vid_path in enumerate(VID_PATHS):

            # Load labels of a vid_path
            self.vid_path = vid_path
            self.load_labels()

            crop_params_frames = {}

            # Iterate frames and cameras
            for frame in self.frames_labeled_indices:
                crop_params = {}
                for cam_name, camera in self.sys_dict.items():
                    cam_name = str(cam_name)

                    # Give a unique image name
                    new_img_name = f'{vid_index}_cam_{cam_name}_{frame}.png'

                    [UL, LR] = self.crop_3d(frame, cam_name, dev_mm=dev_mm)

                    # Load and crop the image
                    img = self.load_labeled_image(frame, cam_name)

                    # Check if cropping dimensions exceed the image
                    # Crop them and change the cropping parameters to no cropping if suggested cropping is outside of
                    # image area
                    cropped_img, [UL, LR] = utils.possible_cropping(img, UL, LR)

                    # Save params in dictionary
                    crop_params[cam_name] = [UL, LR]

                    # Save image into destination folder
                    dest_img_path = join(DEST, new_img_name)
                    cv2.imwrite(dest_img_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

                # Save camera cropping parameters
                crop_params_frames[frame] = crop_params

            crop_vid_params[vid_index] = crop_params_frames

        np.save(join(DEST, "cropping_info_videos.npy"), crop_vid_params)


    def crop_3d(self, frame, cam_name, dev_mm, ind_index=None):
        """
        Crop in 3d along camera plane
        :param frame:
        :param cam_name:
        :param dev_mm:
        :param shift_labels: Shift labels as well? Will be returned as well then
        :return:
        """

        if ind_index is not None:

            pose_3d = self.lift_pose(frame=frame, ransac=True, forced_ind_index=ind_index)[0]
            frame_labels = self.labels[cam_name][ind_index*self.nb_markers:(ind_index+1)*self.nb_markers, :, frame]
        else:
            pose_3d = self.lift_pose(frame=frame, ransac=True)[0]
            frame_labels = self.labels[cam_name][:, :, frame]

        # min_x, max_x, min_y, max_y
        extrems = [np.argmin(frame_labels[:, 0]), np.argmax(frame_labels[:, 0]),
                   np.argmin(frame_labels[:, 1]), np.argmax(frame_labels[:, 1])]
        vec_x = np.array([1, 0, 0])
        vec_y = np.array([0, 1, 0])
        shift_vecs = [-vec_x, vec_x, -vec_y, vec_y]

        # Get camera parameters
        R = self.sys_dict[cam_name].R
        R_w = R.T

        # Copy pose
        pose_altered_3d = copy.deepcopy(pose_3d)

        for index, m_index in enumerate(extrems):
            # Min x shift
            shift_vec_3d = (R_w @ np.expand_dims(shift_vecs[index], axis=0).T)
            pose_altered_3d[m_index] += dev_mm * shift_vec_3d.squeeze()

        new_markers_2d = self.reproject_single(frame, name=cam_name, p_3d=pose_altered_3d)

        extrems_2d_coords = []

        for m_index in extrems:
            extrems_2d_coords.append(new_markers_2d[m_index, :].tolist())

        extrems_2d_coords = np.array(extrems_2d_coords)

        # Floor and ceil to integer
        UL = np.array(
            [math.floor(np.min(extrems_2d_coords[:, 0])), math.floor(np.min(extrems_2d_coords[:, 1]))])
        LR = np.array(
            [math.ceil(np.max(extrems_2d_coords[:, 0])), math.ceil(np.max(extrems_2d_coords[:, 1]))])

        return [UL, LR]


    def write_frame_markers_row(self, writer, coords, labels, image_shape, all_names_ordered, reduce=False, vid_index=0,
                                names_list=[], convert_to_single_animal=True):
        """
        :param writer:
        :param coords:
        :param labels:
        :param image_shape: Shape of image as ndarray
        :param reduce:
        :return:
        """


        h, w = image_shape[:2]
        bp_iter = self.bodyparts_names

        if reduce:
            bp_iter = self.reduced_bp


        # Depending on position of the name in max name list, make empty strings

        # Changed to directly use all names ordered
        #max_names_list = max(names_list)


        name_list_video = names_list[vid_index]

        if not convert_to_single_animal:
            # Init labels to -1 so if there's an individual not labeled, it gets filtered out by empty strings later
            altered_labels = -1 * np.ones(shape=(self.nb_markers*self.nb_individuals, 2))
            for index, name in enumerate(all_names_ordered):
                if name in name_list_video:

                    index_in_video = name_list_video.index(name)

                    # Stack them according to largest list.
                    altered_labels[index*self.nb_markers:(index+1)*self.nb_markers, :] = \
                        labels[index_in_video*self.nb_markers:(index_in_video+1)*self.nb_markers, :]

            labels = altered_labels

        # Iterate the individuals
        for j in range(self.nb_individuals):
            # Iterate the bodyparts
            for bp in bp_iter:
                if bp in self.bodyparts_names:

                    # Gets position of bodypart in our bodypart set
                    index = self.bodyparts_names.index(bp)

                    # Offset index for multiple individuals
                    x = labels[index + j*self.nb_markers, 0]
                    y = labels[index + j*self.nb_markers, 1]

                    # Exclude labels that are not within the image and substitute them by an empty string.
                    if (0 <= x < w) and (0 <= y < h):
                        coords.append(x)
                        coords.append(y)
                    else:
                        coords.append('')
                        coords.append('')

        writer.writerow(coords)


    def alter_DLC_config(self, bodyparts, marker_pairs, config_path, batch_size = 8, PAN=False):


        # Load the config first
        ruamel_file = YAML()
        with open(config_path, "r") as f:
            yaml_file = ruamel_file.load(f)


        yaml_file['batch_size'] = batch_size

        if self.nb_individuals == 1:

            # Set the new bodyparts
            yaml_file['bodyparts'] = bodyparts

        else:
            # Config file changes attributes
            yaml_file['multianimalbodyparts'] = bodyparts

            #ind_list = []
            #for j in range(self.nb_individuals):
            #    ind_list.append(f"individual{j+1}")

            # IN PANOPTIC CASE there is the individuals as list of list passed
            if self.names.__len__() == 1:
                self.names = self.names[0]

            # Also make sure individuals are set
            yaml_file['individuals'] = self.names

        # Set skeleton structure, remains the same for multi-animal
        skeleton_list = []
        for pair in marker_pairs:
            # -1 as 1 indexed
            bone = []
            bone.append(self.bodyparts_names[pair[0] - 1])
            bone.append(self.bodyparts_names[pair[1] - 1])

            skeleton_list.append(bone)

        yaml_file['skeleton'] = skeleton_list

        if PAN:
            yaml_file['identity'] = True

        with open(config_path, "w") as f:
            ruamel_file.dump(yaml_file, f)

        print("Config file has been updated.")


    def export_to_DLC(self, DEST, scorer, subfolder, VID_PATHS, CALIB_PATHS, reduce=False, crop=False, resize=1, dev_mm = 75,
                      convert_to_single_animal=True, PAN=False, name_list=0, image_ext='png', reduced_cam_list=None):
        """

        :param DEST: Labeling destination folder
        :param scorer: name of the scorer specified in DLC project
        :param subfolder: Subfolder in DLC project
        :param VID_PATHS: list of labeled video paths
        :param reduce: Reduce the joints?
        :param crop: crop them from space?
        :return:
        """

        if self.bodyparts_names is None:
            raise ValueError("Please define bodyparts_names for DLC export!")

        if reduce and self.reduced_bp is None:
            raise ValueError("Please define reduced body parts for reduced DLC export!")


        if not PAN:

            names_list = []

            for vid_path in VID_PATHS:

                # Load the config first
                ruamel_file = YAML()
                with open(join(vid_path, 'labeling_info.yaml'), "r") as f:
                    yaml_file = ruamel_file.load(f)

                    names_list.append(yaml_file['names_individuals'])


            # Get video with most individuals
            all_names_ordered = max(names_list)

            # Set nb individuals to largest names list
            self.nb_individuals = len(all_names_ordered)

        else:


            names_list = name_list

            unique_items = []

            for l in name_list:
                for item in l:
                    if item in unique_items:
                        pass
                    else:
                        unique_items.append(item)

            all_names_ordered = unique_items
            self.nb_individuals = len(all_names_ordered)

        # Convert back to one individual if conversion to single animal is needed
        if convert_to_single_animal:
            self.nb_individuals = 1

        # Reduce to OMC bodyparts? in future, make a pass argument
        bp_iter = self.bodyparts_names
        if reduce:
            bp_iter = self.reduced_bp


        with open(join(DEST, f'CollectedData_{scorer}.csv'), 'a', newline='') as f:

            writer = csv.writer(f)

            # Header
            row = ['scorer']
            for i in range(len(bp_iter*self.nb_individuals)):
                row.append('LM')
                row.append('LM')

            writer.writerow(row)

            # Only add individuals row if there are multiple individuals labeled
            if self.nb_individuals > 1:
                row = ['individuals']

                for j in range(self.nb_individuals):
                    for i in range(bp_iter.__len__()):
                        # Twice for x and y, Let individuals be 1-indexed
                        row.append(f"{all_names_ordered[j]}")
                        row.append(f"{all_names_ordered[j]}")

                writer.writerow(row)

            # Write bodyparts
            row = ['bodyparts']

            for j in range(self.nb_individuals):
                # Don't change bodypart names for multiple individuals
                for i in range(len(bp_iter)):
                    row.append(bp_iter[i])
                    row.append(bp_iter[i])

            writer.writerow(row)

            # Write coords definitions
            row = ['coords']
            for i in range(len(bp_iter)*self.nb_individuals):
                row.append('x')
                row.append('y')

            writer.writerow(row)

            # Add this for each row at start
            base_path = "labeled-data\\" + subfolder + '\\'

            # Alter here the vidpath in all vidpaths,
            # load labels and do the same thing

            crop_vid_info = {}

            for index, vid_path in enumerate(VID_PATHS):

                self.vid_path = vid_path
                self.load_labels()

                self.calib_path = CALIB_PATHS[index]

                print(self.calib_path)

                self.read_cam_params()

                if not crop:
                    self.write_video_labels_to_DLC(writer=writer, base_path=base_path, DEST=DEST, vid_index=index, all_names_ordered=all_names_ordered,
                                               reduce=reduce, crop=crop, dev_mm=dev_mm, resize=resize,
                                                   names_list=names_list,
                                                   convert_to_single_animal=convert_to_single_animal, image_ext=image_ext, reduced_cam_list=reduced_cam_list)
                else:
                    print("Crop condition")
                    crop_info = self.write_video_labels_to_DLC(writer=writer, base_path=base_path, DEST=DEST,
                                                               vid_index=index, all_names_ordered=all_names_ordered, reduce=reduce, crop=crop, dev_mm=dev_mm,
                                                               resize=resize, names_list=names_list,
                                                               convert_to_single_animal=convert_to_single_animal, image_ext=image_ext, reduced_cam_list=reduced_cam_list)

                    crop_vid_info[index] = crop_info

            # Save all the aggregated cropping information in one file still.
            np.save(join(DEST, "cropping_info_DLC_export_all.npy"), crop_vid_info)

    def copy_reduced_labels(self, method='single', fraq=None, image_ext='png'):
        """
        Creates a copy of the labels but reduced based on the method for generalization testing
        :param method:
        :return:
        """

        # Select the subset of labeled frames
        nb_labels = self.frames_labeled_indices.__len__()

        subset_labels_indices = []

        if method == 'single':

            subset_labels_indices = np.array(self.frames_labeled_indices[nb_labels//2])

        elif method == 'half':

            subset_labels_indices = self.frames_labeled_indices[::2]

        elif method == 'fraq':
            # Use a fraction of the whole sequence to determine the number of images to retrieve and divide the sequence
            # into by the number of images retrieved uniformly (round wherever needed)


            assert fraq != 0

            assert fraq <= 1

            extract = nb_labels * fraq

            # Extract must be smaller than nb_labels, and as a result make sure to have fraq <=1
            for f in np.arange(0, nb_labels, nb_labels / extract):
                subset_labels_indices.append(round(f))

            subset_labels_indices = self.frames_labeled_indices[subset_labels_indices]

            method = str(fraq)

        # Make a new directory
        reduced_label_dir = join(self.vid_path, method)

        # Copy all labels into it
        reduced_label_label_dir = join(reduced_label_dir, 'labeled_images')
        shutil.copytree(join(self.vid_path, 'labeled_images'), reduced_label_label_dir)
        shutil.copy(join(self.vid_path, "labeling_info.yaml"), join(reduced_label_dir, "labeling_info.yaml"))

        # Set labels for

        labels_reduced = copy.deepcopy(self.labels)
        occs_reduced = copy.deepcopy(self.occs)

        for frame in self.frames_labeled_indices:

            if frame in subset_labels_indices:
                pass
            else:
                for cam in self.sys_dict.keys():
                    #nb_markers, _, nb_frames = labels_reduced[cam].shape

                    labels_reduced[cam][:, :, frame] = np.NaN
                    occs_reduced[cam][:, frame] = 1
                    os.remove(join(join(reduced_label_label_dir, f"cam{cam}"), f"{frame}.{image_ext}"))

        np.save(os.path.join(reduced_label_dir, f"labels.npy"), labels_reduced)
        np.save(os.path.join(reduced_label_dir, f"occs.npy"), occs_reduced)








