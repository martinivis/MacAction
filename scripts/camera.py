"""
Camera and CameraSystem classes for storing OpenCV camera parameters, triangulation,
loading labels and animation.
"""

# Imports
import copy
import cv2
import os
from os.path import join
import numpy as np
import scripts.utils as utils
import matplotlib.pyplot as plt
import scipy.io as sio
import toml
import matplotlib
import json
import matplotlib.animation


# to stop interactive mode
matplotlib.use('Qt5Agg')

class System:

    def __init__(self, calib_path, nb_individuals=1):
        """
        Initialise the system object.
        :param calib_path: Path to the calibration file.
        :param nb_individuals: Number of individuals to track potentially
        """

        # Dictionary containing cameras by names as keys
        self.sys_dict = {}
        self.calib_path = calib_path

        # Video labels path
        self.vid_path = None
        self.vid_length = 0

        # system labels
        # Dictionary over camera names: with (nb_markers, 2, nb_frames_labeled) and pixel positions
        self.labels = {}
        # Dictionary over camera names: with (nb_markers, nb_frames_labeled) and occlusion or not as 0 (vis) or 1 (occ)
        self.occs = {}

        # If labels and points should be disturbed for calibration analysis
        self.disturb = False
        # GT Labels that are undisturbed
        self.gt_labels = {}

        # Labeled frames
        self.frames_labeled_indices = None
        self.frames_labeled = {}

        # Triangulated poses
        self.poses = {}
        # Reprojections and errors
        self.reprojections = {}
        self.reprojection_ers = {}
        # Triangulation camera sets, only for ransac
        self.best_sets_frames = {}
        self.triang_dic_frames = {}

        # Number of individuals
        self.nb_individuals = nb_individuals

        # Visualization specific
        self.scatter3d = None
        self.title_mocap = None
        self.lines_mocap = None
        self.conf_threshold = 0.1
        self.show_skeleton = True


        # Smaller marker pair set of macaques
        self.marker_pairs = ((1, 2), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6), (6, 9), (9, 10), (9,11), (9,12), (12,13), (12,19),
                             (13, 14), (14, 15), (15,16), (15,17), (17,18),
                             (19, 20), (20, 21), (21, 22), (21, 23), (23,24), (12,25), (25, 36), (36, 26), (36,31),
                             (26, 27), (27,28),(28,29),(28,30),
                             (31, 32), (32, 33),(33, 34), (33, 35), (36,37), (37,38))


        # For animation
        self.title_prefix = None
        self.snapshot = 0
        self.limb_constraints = None
        self.with_ransac = False

    def init_default_labels(self, nb_individuals=1):
        # Create an empty array of nb markers and vid_length
        empty_array = np.empty(shape=(nb_individuals*self.nb_markers, 2, self.vid_length))
        empty_array[:] = np.NaN

        for frame in self.frames_labeled_indices:
            empty_array[:, :, frame] = np.zeros(shape=(nb_individuals*self.nb_markers, 2))

        # Init every marker as occluded first and then change to visible when touched
        occs_init = np.ones(shape=(nb_individuals*self.nb_markers, self.vid_length))

        for cam in self.sys_dict.keys():
            self.labels[cam] = copy.deepcopy(empty_array)
            self.occs[cam] = copy.deepcopy(occs_init)

    def update_3d_animation(self, num):
        """
        Update function for animation
        :param num:
        :return:
        """
        # Get 3d pose
        p_3d = self.poses[num]
        p_3d = p_3d[:self.nb_individuals * self.nb_markers, :]

        # Change the position of the scattered points
        self.scatter3d._offsets3d = (p_3d[:, 0], p_3d[:, 1], p_3d[:, 2])
        # Set the new title
        self.title_mocap.set_text(self.title_prefix + '3D Pose, frame={}'.format(num))

        if num % 1000 == 0 and num != 0:
            print("A thousand frames done")

        if self.show_skeleton:
            # Draw line segments of new points
            for ind in range(self.nb_individuals):
                for line_index, pair in enumerate(self.marker_pairs):
                    (m1, m2) = pair

                    m1 = m1 + ind * self.nb_markers - 1
                    m2 = m2 + ind * self.nb_markers - 1

                    line = self.lines_mocap[line_index + ind * self.marker_pairs.__len__()]

                    # If both markers of segment do not reside in origin, draw them, else make no line (from 0 to 0)
                    if not (np.sum(p_3d[m1]) == 0 or np.sum(p_3d[m2]) == 0):
                        line.set_data([p_3d[m1, 0], p_3d[m2, 0]], [p_3d[m1, 1], p_3d[m2, 1]])
                        line.set_3d_properties([p_3d[m1, 2], p_3d[m2, 2]])
                    else:
                        # Make no line at all
                        line.set_data([0, 0], [0, 0])
                        line.set_3d_properties([0, 0])

        return self.scatter3d, self.lines_mocap, self.title_mocap,

    def show_3d_poses_as_video(self, lims=[[-600, 400],[-400, 600],[-200, 500]], show_skeleton=True,
                               length=None, title_prefix="", set_lims=True):
        """
        Animation function of previously loaded poses
        """

        # Init figure and ax
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        if set_lims:
            ax.set_xlim(lims[0][0], lims[0][1])
            ax.set_ylim(lims[1][0], lims[1][1])
            ax.set_zlim(lims[2][0], lims[2][1])

        # Get initial 3d pose
        p_3d = self.poses[0]

        p_3d = p_3d[:self.nb_individuals * self.nb_markers, :]

        # Scatter markers and save in class
        markers = ax.scatter(p_3d[:, 0], p_3d[:, 1], p_3d[:, 2])
        self.scatter3d = markers

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Scatter cameras in plot
        for cam_key, cam in self.sys_dict.items():
            R_abs, t_abs = utils.rel_abs_conversion(cam.R, cam.t)
            t_abs = np.squeeze(t_abs)

            ax.scatter(t_abs[0], t_abs[1], t_abs[2], s=20, c='k')
            ax.text(t_abs[0], t_abs[1], t_abs[2], f"{cam_key}", color='k')

        # Make title and save in class
        self.title_prefix = title_prefix
        title = ax.set_title(title_prefix)
        self.title_mocap = title

        self.show_skeleton = show_skeleton

        if show_skeleton:
            # Make line segments and save in class, if either of the markers resides in the origin don't draw the segment
            lines = []

            for ind in range(self.nb_individuals):
                for line_index, pair in enumerate(self.marker_pairs):
                    (m1, m2) = pair

                    m1 = m1 + ind*self.nb_markers - 1
                    m2 = m2 + ind * self.nb_markers - 1
                    #m1 -= 1
                    #m2 -= 1
                    if not (np.sum(p_3d[m1]) == 0 or np.sum(p_3d[m2]) == 0):
                        line = ax.plot([p_3d[m1, 0], p_3d[m2, 0]], [p_3d[m1, 1], p_3d[m2, 1]], [p_3d[m1, 2], p_3d[m2, 2]])
                    else:
                        line = ax.plot([0, 0], [0, 0], [0, 0])
                    # [0] to get the artist itself
                    lines.append(line[0])
            self.lines_mocap = lines

        duration = len(self.poses)

        if length is not None:
            duration = length

        # Animation function of matplotlib
        ani = matplotlib.animation.FuncAnimation(fig, self.update_3d_animation, duration,
                                                 interval=40, blit=False)

        if self.limb_constraints is not None:
            snap_path = join(self.vid_path, f"{self.snapshot}")
            anim_path = join(snap_path, f"Limb_constr_{self.limb_constraints}")

            try:
                os.mkdir(anim_path)
            except:
                print(f"Specific animation folder already exists.")

            # Save animation
            ani.save(join(anim_path, f"mocap_{self.conf_threshold}_{self.limb_constraints}_{self.with_ransac}_"
                                     f"skel{self.show_skeleton}.gif"))

        else:

            # Save the animation
            ani.save(join(self.vid_path,  f"mocap_as_video.gif"))

        # Close the figure
        plt.close(fig)

        print("Animation done!")

    def load_labeled_image(self, frame, cam_name, extension = 'png'):
        """
        Load labeled image for a given frame and camera (by name)
        """
        img = cv2.imread(os.path.join(self.vid_path, rf'labeled_images\cam{cam_name}\{frame}.{extension}'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def show_labeled_image(self, frame, name, m2_r=None, text=None):
        """
        Display a labeled image
        """
        img = self.load_labeled_image(frame, name)
        label = self.labels[name][:, :, frame]
        occ = self.occs[name][:, frame]

        # Visible and non-visible markers of labeled image
        vis = label[occ == 0][:]
        n_vis = label[occ == 1][:]

        plt.figure()
        plt.imshow(img)
        plt.scatter(vis[:, 0], vis[:, 1], 3, label='Visible')
        # If the reprojections are done automatically they will overlap with the n_vis.
        plt.scatter(n_vis[:, 0], n_vis[:, 1], 3, label='Occ')

        if text:
            for j in range(self.nb_individuals):
                for m in range(self.nb_markers):
                    m += j*self.nb_markers
                    plt.text(label[m, 0], label[m, 1], '%s' % (str(m + 1)), size=10, zorder=1, color='k')
                    if m2_r is not None:
                        plt.text(m2_r[m, 0], m2_r[m, 1], '%s*' % (str(m + 1)), size=10, zorder=1, color='k')
        if m2_r is not None:
            plt.scatter(m2_r[:, 0], m2_r[:, 1], 3, label='Re-projected')

        # Only works then if previously all markers were lifted.
        if self.reprojection_ers.__len__():
            title = f"Mean marker reprojection for frame {frame} in camera " \
                    f"{name}: {self.reprojection_ers[frame][name]['mean_points']}"

        else:
            title = f"Frame {frame} with (reprojected) labels in camera {name}"

        plt.title(title)
        plt.legend()
        plt.show()


    def lift_marker(self, vis_labels, undist=True, ransac=False, orig_labels=None, min_nb_triang=2, max_nb_triang=0,
                    ransac_minimal=True):
        """

        :param vis_labels: Dictionary of cameras with a particularly visible 2d marker for that camera
        :param undist: Undistort labels
        :param ransac: Use sets of cameras for minimal reprojection error
        :param orig_labels: For calibration analysis
        :param min_nb_triang: Minimum number of cameras in a set
        :param max_nb_triang: Maximum number of cameras in a set
        :param ransac_minimal: Reduce ransac output only to the 3D estimate
        :return: X (3, ) best estimate, best index, triang dic with reprojection errors
        """

        points = {}
        cam_Ms = {}

        for index, name in enumerate(vis_labels.keys()):
            coords = vis_labels[name]
            cam = self.sys_dict[name]

            # Takes already cam K into account
            if undist and ransac is False:
                coords = cv2.undistortPoints(coords.copy(), cameraMatrix=cam.K, distCoeffs=cam.d)[0, 0, :]


            points[name] = coords

            M = np.vstack((np.hstack((cam.R, cam.t)), np.array([0, 0, 0, 1])))
            cam_Ms[name] = M

        # Order should be preserved in both dictionaries
        if ransac:
            # Best 3d coordinates, index in triang_dic of best 3d coords, and triang dict
            X, best_index, triang_dic = self.ransac_triangulation(points, cam_Ms, orig_labels, min_nb_triang=min_nb_triang, max_nb_triang=max_nb_triang)


            if ransac_minimal:
                # return best 3d, return its mean reprojection, return the set of the best triang, and triang itself
                return X, 0, 0, 0, 0
            else:
                # return best 3d, return its mean reprojection, return the set of the best triang, and triang itself
                return X, np.array(list(triang_dic[best_index][2].values())).mean(), triang_dic[best_index][0], best_index,\
                       triang_dic

        else:
            # approx 35us for one triangulation
            X = utils.triangulate_simple(list(points.values()), list(cam_Ms.values()))

            er_list = []
            for cam_name in vis_labels.keys():
                cam = self.sys_dict[cam_name]
                rvec = cv2.Rodrigues(cam.R, None, None)[0]
                x_2d_rep = np.squeeze(cv2.projectPoints(X, rvec, cam.t, cam.K, cam.d)[0], axis=1)

                if not self.disturb:
                    e = utils.rep_error(x_2d_rep, vis_labels[cam_name])
                else:
                    e = utils.rep_error(x_2d_rep, orig_labels[cam_name])

                er_list.append(e)

            return X, np.array(er_list).mean(), False, False, False

    def ransac_triangulation(self, dist_points, Ms, orig_points=None, min_nb_triang=2, max_nb_triang=0):
        """
        Implementation of Ransac triangulation for retrieving minimal reprojection error for a given set
        :param dist_points: Distorted points
        :param Ms: World to camera transform
        :param orig_points: Orig points in case of calibration analysis
        :param min_nb_triang: Minimum number of cameras in a set
        :param max_nb_triang: Maximum number of cameras in a set
        :return:
        """
        # Get all possible triangulation sets
        sets = utils.powerset(dist_points, min_nb_triangulation=min_nb_triang, max_set=max_nb_triang)

        triang_dict = {}

        for index, set in enumerate(sets):

            point_list = []
            M_list = []

            for cam_name in set:
                cam = self.sys_dict[cam_name]

                undist_point = cv2.undistortPoints(dist_points[cam_name].copy(),
                                                   cameraMatrix=cam.K, distCoeffs=cam.d)[0, 0, :]
                point_list.append(undist_point)
                M_list.append(Ms[cam_name])

            X = utils.triangulate_simple(point_list, M_list)

            er_list = {}

            # Calculate reprojections and errors for all cameras!!!
            for cam_name in dist_points.keys():

                cam = self.sys_dict[cam_name]
                rvec = cv2.Rodrigues(cam.R, None, None)[0]
                x_2d_rep = np.squeeze(cv2.projectPoints(X, rvec, cam.t, cam.K, cam.d)[0], axis=1)

                if not self.disturb:
                    e = utils.rep_error(x_2d_rep, dist_points[cam_name])
                else:
                    e = utils.rep_error(x_2d_rep, orig_points[cam_name])

                er_list[cam_name] = e

            triang_dict[index] = [set, X, er_list]

        # Get best mean reprojection error

        # If there are no sets, return 3d point at [-1, -1, -1]
        if len(sets):
            best_ = X

            best_index = 0
            # Default error
            prev_error = 1000

            for key, value in triang_dict.items():

                er_s = list(value[2].values())
                mean = np.array(er_s).mean()

                if mean < prev_error:
                    best_ = value[1]
                    best_index = key
                    prev_error = mean
        else:
            best_ = -1 * np.ones(3)
            best_index = -1


        return best_, best_index, triang_dict



    def pose_dict_to_point_tracks(self, marker_dicts, frame):
        # Analysis of calibration
        table = -1 * np.ones(shape=(1, self.sys_dict.__len__(), marker_dicts.__len__(), 2))

        for key, value in marker_dicts.items():
            # Key is the marker id, value the cameras that observed it in a vis_m_dict
            # Iter all camera names
            for index, cam in enumerate(self.sys_dict.keys()):
                if cam in value.keys():
                    # point is already in 1x2
                    table[0, index, key] = value[cam]

        pt_obj = np.array(self.gen_point_tracks(table), dtype=np.object)
        sio.savemat(os.path.join(self.vid_path, f'pose_{frame}_point_tracks.mat'), {'pt_tr': pt_obj})

    def lift_pose(self, frame, show=False, save_as_point_track=False, undistort=True, ransac=False, verbose=False,
                  save_pose_npy=False, forced_ind_index=None, min_nb_triang=2, max_nb_triang=0):
        """
        Lift the pose of a specific frame
        :param frame: The frame to pose
        :param show: Display the pose?
        :param save_as_point_track: Save in track format?
        :param undistort: Undistort points
        :param ransac: Ransac or simple DLT?
        :param verbose: Show process
        :param save_pose_npy: Save as npy?
        :param forced_ind_index: Change individual index
        :param min_nb_triang: Minimum number of cameras in a set
        :param max_nb_triang: Maximum number of cameras in a set
        :return:
        """
        p_3d = np.zeros((self.nb_markers*self.nb_individuals, 3))
        marker_dicts = {}

        rep_ers = []
        best_sets = {}
        triang_dicts = {}

        # If you use forced individual make sure to have nb_individuals set to 1
        for j in range(self.nb_individuals):

            if forced_ind_index is not None:
                j = forced_ind_index

            for m in range(self.nb_markers):
                # Create visible dictionary for a single marker
                vis_m_dict = {}

                # Shift labels if we're looking at another individual
                m_shift = m + self.nb_markers * j
                if self.disturb:
                    orig_m_dict = {}

                for cam_name in self.sys_dict.keys():
                    if self.occs[cam_name][m_shift, frame] == 0:
                        vis_m_dict[cam_name] = self.labels[cam_name][m_shift, :, frame]
                        if self.disturb:
                            orig_m_dict[cam_name] = self.gt_labels[cam_name][m_shift, :, frame]

                # Lift if there are more than 2 views
                if vis_m_dict.__len__() > 1:

                    if not self.disturb:
                        r, rep_error, best_set, best_index, triang_dic = self.lift_marker(vis_labels=vis_m_dict,
                                                                              undist=undistort, ransac=ransac, min_nb_triang=min_nb_triang, max_nb_triang=max_nb_triang)
                    else:
                        r, rep_error, best_set, best_index, triang_dic = self.lift_marker(vis_labels=vis_m_dict,
                                                                              undist=undistort, ransac=ransac,
                                                                              orig_labels=orig_m_dict, min_nb_triang=min_nb_triang, max_nb_triang=max_nb_triang)

                    marker_dicts[m_shift] = vis_m_dict
                    rep_ers.append(rep_error)
                    best_sets[m_shift] = best_set
                    triang_dicts[m_shift] = triang_dic
                else:
                    r = np.zeros((3))
                    #print(f"!Marker {m + 1}!, Nb_visible views: {vis_m_dict.__len__()}")

                if forced_ind_index is not None:
                    p_3d[m] = r
                else:
                    p_3d[m_shift] = r

        # Save already that pose into dictionary in case you want to reproject it without lifting all labels into a
        # particular image
        self.poses[frame] = p_3d


        if verbose:
            for m in range(self.nb_markers):
                print("Marker: ", m + 1)
                print("Rep_error: ", rep_ers[m])
                print(triang_dicts[m])

            #print("Triang dicts: ", triang_dicts)

        mean_rep_pose = -1

        if rep_ers:
            mean_rep_pose = np.array(rep_ers).mean()

        if save_as_point_track:
            self.pose_dict_to_point_tracks(marker_dicts, frame)

        if show:
            self.show_pose_3d(p_3d, frame, mean_rep_pose)

        if save_pose_npy:
            np.save(os.path.join(self.vid_path, f"single_pose_{frame}.npy"), p_3d)

        return p_3d, mean_rep_pose, best_sets, triang_dicts

    def load_labels(self):
        # Load labels from files
        labels = np.load(os.path.join(self.vid_path, "labels.npy"), allow_pickle=True)[()]
        self.labels = labels
        self.occs = np.load(os.path.join(self.vid_path, "occs.npy"), allow_pickle=True)[()]

        exemplary_cam_labels = self.labels[list(self.sys_dict.keys())[0]]
        self.frames_labeled_indices = np.nonzero(np.sum(np.sum(~np.isnan(exemplary_cam_labels),
                                                               axis=0), axis=0))[0]
        self.vid_length = exemplary_cam_labels.shape[-1]
        if self.disturb:
            self.gt_labels = copy.deepcopy(labels)

    def show_pose_3d(self, m_3d, frame, mean_marker_rep_error, show_skel=True):
        # Show pose in 3d
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(m_3d[:, 0], m_3d[:, 1], m_3d[:, 2])

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.grid(False)

        for m in range(self.nb_markers*self.nb_individuals):
            ax.text(m_3d[m, 0], m_3d[m, 1], m_3d[m, 2], '%s' %
                    (str((m % self.nb_markers)+1)), size=10, zorder=1, color='k')

        for pair in self.marker_pairs:

            for ind_index in range(self.nb_individuals):

                (m1, m2) = pair
                m1 -= 1
                m2 -= 1

                m1 += ind_index*self.nb_markers
                m2 += ind_index*self.nb_markers

                if not (np.sum(m_3d[m1]) == 0 or np.sum(m_3d[m2]) == 0):
                    if show_skel:
                        ax.plot([m_3d[m1, 0], m_3d[m2, 0]], [m_3d[m1, 1], m_3d[m2, 1]], [m_3d[m1, 2], m_3d[m2, 2]])


        plt.title(f"Pose for frame {frame} with mean marker rep error {mean_marker_rep_error}")
        plt.show()


    def reproject_single(self, frame, name, p_3d = None, show=False, text=True):
        """
        Reproject a single 3d point into a specific camera view
        :param frame: The frame to reproject
        :param name: Name of the camera
        :param p_3d: The 3D point
        :param show: Display?
        :param text: Text for display
        :return: m2d_r of shape 1x2
        """
        cam = self.sys_dict[name]
        rvec = cv2.Rodrigues(cam.R, None, None)[0]
        if p_3d is not None:
            #print("Using the new points given")
            m2d_r = np.squeeze(cv2.projectPoints(p_3d, rvec, cam.t, cam.K, cam.d)[0], axis=1)
        else:
            p_3d = self.poses[frame]
            m2d_r = np.squeeze(cv2.projectPoints(p_3d, rvec, cam.t, cam.K, cam.d)[0], axis=1)

            # if in that frame camera has markers calculate the rep error.

            # m2d_r shape is nb marker 2
            # because we init 3d points with (0,0,0) need to be excluded for errors
            m2d_r[np.where(p_3d.sum(axis=1) == 0), :] = 0
        if show:
            self.show_labeled_image(frame, name, m2d_r, text)
            return True

        return m2d_r

    def read_cam_params(self, alter_path=None, extension = "sys_calib"):
        """
        Read calibration parameters from calibration file
        Either toml format or matlab format, also checking potential re-calibration files
        :param alter_path: An alternative path to load the calibration
        :param extension: Different naming of the matlab file?
        :return:
        """

        if 'toml' in self.calib_path:
            self.read_cam_params_from_toml(base_path=self.calib_path)
            return True
        if alter_path is not None:
            obj_array = sio.loadmat(os.path.join(alter_path, extension + '.mat'))
        else:
            if '.json' in self.calib_path:
                self.read_cam_params_from_json(self.calib_path)
                return True

            else:
                try:
                    obj_array = sio.loadmat(os.path.join(self.calib_path, extension + '.mat'))
                except OSError:
                    print("There was no system calibration file in ", self.calib_path)
                    try:
                        print("Try loading a recalibration file")
                        obj_array = sio.loadmat(os.path.join(self.calib_path, 'recalibration.mat'))
                    except OSError:
                        raise ValueError("No calibration files found in the specified path")

        obj_array = obj_array["Cameras"][0]

        for i in range(obj_array.__len__()):

            cam = Camera()
            name, cam.im_size, cam.K, cam.d, cam.R, cam.t, cam.calib_error = obj_array[i][0]
            cam.d = cam.d.squeeze()
            cam.im_size = cam.im_size[0]
            self.sys_dict[name[0]] = cam

        # Sort alphabetically
        self.sys_dict = dict(sorted(self.sys_dict.items()))
        print("Parameters loaded.")

    def read_cam_params_from_toml(self, base_path):
        """
        Read the camera parameters from toml file
        :param base_path: Path of the calibration toml file
        :return:
        """

        if '.toml' in base_path:
            cam_file = toml.load(base_path)
        else:
            cam_file = toml.load(os.path.join(base_path, "calibration.toml"))

        for index, cam_key in enumerate(dict.keys(cam_file)):
            if cam_key != 'metadata':
                cam = Camera()
                name = cam_file[cam_key]['name']
                cam.K = np.array(cam_file[cam_key]['matrix'])
                cam.d = np.array(cam_file[cam_key]['distortions'])
                cam.r = np.array(cam_file[cam_key]['rotation'])
                cam.R = cv2.Rodrigues(cam.r)[0]
                # expand_dims so it is 3x1
                cam.t = np.expand_dims(np.array(cam_file[cam_key]['translation']), axis=1)
                cam.im_size = np.array(cam_file[cam_key]['size'])
                self.sys_dict[name] = cam

        # Sort alphabetically
        self.sys_dict = dict(sorted(self.sys_dict.items()))

    def read_cam_params_from_json(self, json_path):
        """
        Read the camera parameters from json
        :param json_path: Json path
        :return:
        """
        with open(json_path) as f:
            d = json.load(f)
        cameras_json = d["Calibration"]["cameras"]

        for cam_index in range(len(cameras_json)):
            camera = cameras_json[cam_index]

            intrinsics = camera['model']['ptr_wrapper']['data']

            # Make it numpy arrays
            name = cam_index + 1

            im_height = intrinsics['CameraModelCRT']['CameraModelBase']['imageSize']['height']
            im_width = intrinsics['CameraModelCRT']['CameraModelBase']['imageSize']['width']

            # Get intrinsic parameters
            ar = intrinsics['parameters']['ar']['val']
            cx = intrinsics['parameters']['cx']['val']
            cy = intrinsics['parameters']['cy']['val']

            f = intrinsics['parameters']['f']['val']
            k1 = intrinsics['parameters']['k1']['val']
            k2 = intrinsics['parameters']['k2']['val']
            k3 = intrinsics['parameters']['k3']['val']

            p1 = intrinsics['parameters']['p1']['val']
            p2 = intrinsics['parameters']['p2']['val']

            # Intrinsic camera parameters
            intr = np.eye(3)

            intr[0, 0] = f
            intr[1, 1] = ar * f
            intr[0, 2] = cx
            intr[1, 2] = cy

            # Distortion parameters
            dist = np.zeros(5)
            dist[0] = k1
            dist[1] = k2
            dist[-1] = k3

            dist[2] = p1
            dist[3] = p2

            # rot = camera['transform']['rotation']
            rot_as_array = np.array(list(camera['transform']['rotation'].values()))
            # pos = camera['transform']['translation']
            pos_as_array = np.array(list(camera['transform']['translation'].values()))

            cam = Camera()
            name = str(name)
            cam.K = intr
            cam.d = dist
            cam.r = rot_as_array

            cam.R = cv2.Rodrigues(cam.r, None, None)[0]
            # expand_dims so it is 3x1
            # Convert to mm instead of m
            cam.t = np.expand_dims(pos_as_array, axis=-1) * 1000
            cam.im_size = np.array([im_height, im_width])
            self.sys_dict[name] = cam

        # Sort alphabetically
        self.sys_dict = dict(sorted(self.sys_dict.items()))


    def export_cam_params_as_toml(self, base_path):
        """
        Export the camera parameters to toml
        :param base_path: Path to export to
        :return:
        """

        save_dict = {}

        for index, cam_key in enumerate(self.sys_dict.keys()):

            camera = self.sys_dict[cam_key]

            param_dict = {}

            param_dict['name'] = str(cam_key)

            # h, w needs to be reversed?
            h, w = camera.im_size
            param_dict['size'] = np.array([w, h])

            param_dict['matrix'] = camera.K
            param_dict['distortions'] = camera.d

            # check if r needs to be computed
            r = 0

            if camera.r is None:
                r = cv2.Rodrigues(camera.R)[0]
            else:
                r = camera.r

            param_dict['rotation'] = np.squeeze(r)

            param_dict['translation'] = np.squeeze(camera.t)

            save_dict[f"cam_{index}"] = param_dict

        default_meta_data = {}
        default_meta_data["adjusted"] = False
        default_meta_data["error"] = 0.27153887337842425

        save_dict["metadata"] = default_meta_data
        print(base_path)
        with open(join(base_path, "calibration.toml"), 'w') as f:
            toml.dump(save_dict, f, encoder=toml.TomlNumpyEncoder())

    def save_labels(self, alter_string = ""):
        # Save the labels
        np.save(os.path.join(self.vid_path, f"labels{alter_string}.npy"), self.labels)
        np.save(os.path.join(self.vid_path, f"occs{alter_string}.npy"), self.occs)

class Camera:
    def __init__(self):

        # OpenCV specific placeholders
        self.name = None
        self.K = None
        self.d = None
        self.r = None
        self.t = None
        self.R = None
        self.calib_error = None
        self.im_size = None

    def get_rot_mat(self):
        # Return the extrinsic rotation matrix instead of axis-angle for that camera
        return cv2.Rodrigues(self.r)