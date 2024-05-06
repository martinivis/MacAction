import os
import cv2
import numpy as np
import errno
from itertools import chain, combinations
from os.path import join

def rep_error(p1, p2):
    """ Reprojection error of one 'marker'
    :param p1: 1x2 np array
    :param p2: 1x2 np array
    :return:
    """
    return np.linalg.norm(p1-p2)

def rel_abs_conversion(R, t):
    return R.T, -R.T@t

def powerset(iterable, min_nb_triangulation = 2, max_set = 0):
    """
    Adapted from https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    :param iterable:
    :return:
    """
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"

    s = list(iterable)

    if not max_set:
        max_set = s.__len__()

    list_subsets = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    # Convert tuples to lists
    list_subsets = [list(x) for x in list_subsets]
    # Reduce list by min points for triangulation
    list_subsets = [x for x in list_subsets if max_set >= x.__len__() >= min_nb_triangulation]

    return list_subsets

def essential_FromRt(R1, t1, R2, t2):
    """

    :param R1: camera rotation in first camera coordinate system
    :param t1: world origin in first camera coordinate system
    :param R2:
    :param t2:
    :return:
    """

    # Translation of second camera center to first in second's perspective
    t_2_1 = R2@(-R1.T@t1)+t2
    R = R2@R1.T

    # Matrix multiplication of cross product
    E = skew(t_2_1.flatten()) @ R

    return E

def fundamental_From_Essential(K1, K2, E):
    """
    Compute fundamental matrix from essential, and intrinsics, also clean it up so it has guaranteed rank 2
    :param K1:
    :param K2:
    :param E:
    :return:
    """

    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # Clean up of F, so it has rank 2
    u, d, vh = np.linalg.svd(F, full_matrices=True)
    d[2] = 0
    clean_F = u @ np.diag(d) @ vh

    return clean_F

def fundamental_From_RtK(R1, t1, K1, R2, t2, K2):
    E = essential_FromRt(R1, t1, R2, t2)
    return fundamental_From_Essential(K1, K2, E)
def distort_point(x_und, y_und, K, d):
    """
    Distort point after being projected from world to image by K[R|t]
    :param x_und:
    :param y_und:
    :param K: Intrinsics, 3x3
    :param d: Distortion parameters, 5x1 (usually)
    :return: distorted points 1x2?
    """
    # Unpack dist coefficients
    k1, k2, p1, p2, k3 = d

    invK = np.linalg.inv(K)
    z = np.array([x_und, y_und, 1])
    nx = invK@z

    x = nx[0]
    y = nx[1]
    r_sq = x**2 + y**2

    # For radial distortion
    gamma = (1 + k1 * r_sq + k2 * r_sq ** 2 + k3 * r_sq ** 3)

    # First part is radial until +, then tangential
    x_dn = gamma * x + (2 * p1 * x * y + p2 * (r_sq + 2 * x ** 2))
    y_dn = gamma * y + (2 * p2 * x * y + p1 * (r_sq + 2 * y ** 2))

    z2 = np.array([x_dn, y_dn, 1])
    x_d = K.dot(z2)

    return np.array([x_d[0], x_d[1]])

def undist_points(x, K, d):
    """
    Point in distorted image gets undistorted
    :param x: image point/s as Nx1x2
    :param K:
    :param d:
    :return: Undistorted point in homogeneous coordinates
    """

    undist = cv2.undistortPoints(x, cameraMatrix=K, distCoeffs=d)[0, 0, :]
    hom_coords = np.expand_dims(np.array([undist[0], undist[1], 1]), axis=1)
    undist_points = K @ hom_coords

    return undist_points

def line_params(F, undist_point):
    """
    Calculate line parameters in homogeneous coordinates, not taking order into account, 'right' image computation
    Fx_1 and points x_2^T are on the line.
    :param F:
    :param undist_point:
    :return: line parameters
    """

    return (F @ undist_point).flatten()


def adjust_gamma(image, gamma=0.6):
    """
    Adjust the gamma of an image
    :param image:
    :param gamma:
    :return:
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    #invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)



def get_cdf_for_HE(img_brightness):

    # Get brignthness transform
    hist, bins = np.histogram(img_brightness.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    return cdf

def apply_HE_to_BGR_cv_img(img_as_BGR, cdf):
    img_as_YCbCr = cv2.cvtColor(img_as_BGR, cv2.COLOR_BGR2YCrCb)
    img_brightness = img_as_YCbCr[: , :, 0]
    img_as_YCbCr[:, :, 0] = cdf[img_brightness]

    return cv2.cvtColor(img_as_YCbCr, cv2.COLOR_YCrCb2BGR)

def possible_cropping(img, UL, LR):

    h, w = img.shape[:2]

    x_s = 0
    x_e = 0
    y_s = 0
    y_e = 0

    if UL[0] < 0 or UL[0] > w:
        x_s = 0
    else:
        x_s = UL[0]
    if LR[0] < 0 or LR[0] > w:
        x_e = w
    else:
        x_e = LR[0]
    if UL[1] < 0 or UL[1] > h:
        y_s = 0
    else:
        y_s = UL[1]
    if LR[1] < 0 or LR[1] > h:
        y_e = h
    else:
        y_e = LR[1]

    UL = np.array([x_s, y_s])
    LR = np.array([x_e, y_e])

    return img[y_s:y_e, x_s:x_e, :], [UL, LR]

def adjust_videos_per_path(path, gamma=0.6, resize=1, lossy=True, max_frames = None, apply_HE=False,
                           folder_name ="processed", resize_area = None, video_extension = '.avi'):
    """
    Makes videos the same length, changes the gamma value and compression!
    You actually can not compress with IoI first and then load and save uncompressed because this would actually apply
    compression again. Normally you should be able to have it in the same compression format, but hard to handle.
    :param path:
    :param gamma:
    :param lossy:
    :return:
    """
    print(path)
    files = os.listdir(path)

    # ONLY AVI supported because of double compression..
    files = [f for f in files if video_extension in f]

    print(files)
    frames_dict = {}
    frames_list = []

    # Open video files and get common number of images
    for f in files:
        vid = cv2.VideoCapture(os.path.join(path, f))

        frames_dict[f] = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_list.append(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_list = np.array(frames_list)

    min_frames = int(frames_list.min())
    print(frames_dict)
    print(f"Minimum number of frames {min_frames}")


    if resize != 1:
        folder_name = f"resized_{resize}_{apply_HE}"

    try:
        os.mkdir(os.path.join(path, folder_name))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    for f in files:
        vid = cv2.VideoCapture(os.path.join(path, f))

        if (vid.isOpened()== False):
            print("Error opening video stream or file")

        nb_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_width = int(vid.get(3))
        frame_height = int(vid.get(4))

        if resize != 1:

            if resize_area is not None:
                frame_width = resize_area[0]
                frame_height = resize_area[1]
            else:
                frame_width = int(frame_width//resize)
                frame_height = int(frame_height//resize)

            print(f"Frame_height = {frame_height}")
            print(f"Frame_width = {frame_width}")


        if lossy:
            out = cv2.VideoWriter(join(join(path, folder_name), f[:-4] + ".avi"),
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
        else:
            # Uncompressed
            out = cv2.VideoWriter(join(join(path, folder_name), f[:-4] + ".avi"), 0
                                  , fps, (frame_width, frame_height))

        f_c = 0
        cdf = 0

        while(True):

            ret, frame = vid.read()

            if ret == True:

                # If the video got more frames than the minimum videos, skip the initial frames as long as there is
                # the same number of frames
                if nb_frames > min_frames:
                    nb_frames -= 1
                    continue

                else:

                    if gamma != 1:
                        frame = adjust_gamma(frame, gamma=gamma)

                    # Reverse order, so interpolation of intensities matches frame export from labeling
                    # Resizing interpolates already which will cause more compression when using jpeg
                    if resize != 1:

                        if resize_area is not None:
                            frame = cv2.resize(frame, (resize_area[0], resize_area[1]),
                                               interpolation=cv2.INTER_AREA)
                        else:

                            frame = cv2.resize(frame, (int(frame.shape[1] // resize),
                                                       int(frame.shape[0] // resize)),
                                            interpolation=cv2.INTER_AREA)

                    if f_c == 0 and apply_HE:
                        cdf = get_cdf_for_HE(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0])


                    if apply_HE:
                        frame = apply_HE_to_BGR_cv_img(frame, cdf)

                    # Write the frame into the file 'output.avi'
                    out.write(frame)
                    f_c += 1
            else:
                break

        assert min_frames == f_c

        # Release video captures
        vid.release()
        out.release()

    with open(join(join(path, folder_name), "processing_details.txt"), "w") as text_file:

        if lossy:
            text_file.write(f"Compression: MJPG \n")
        else:
            text_file.write(f"Uncompressed \n")

        if resize != 1:
            text_file.write(f"Resizing: {resize} \n")

        text_file.write(f"Gamma: {gamma}")


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d
