from skimage.feature import local_binary_pattern
import numpy as np
from skimage.feature import greycomatrix
import skimage.io as io
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import FLAGS
import keeper
import Preprocessing as pre
import datetime
import os
import Utilities

features_dict = {}
for method in FLAGS.CACHE_FEATURE_VECTORS.keys():
    if FLAGS.CACHE_FEATURE_VECTORS[method]:
        features_dict[method] = keeper.get_tensor_list_dict_from_disk(FLAGS.FEATURE_VECTORS_PATH[method])
    else:
        features_dict[method] = {}


def store_feature_vectors_of_list(form_filenames_list, method=0, log_filename=None):
    if log_filename is None:
        log_filename = 'feature_vector' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '.txt'

    keeper.store_tensor_list(form_filenames_list,
                             lambda name: [getFeatureVector(b, method) for b in pre.get_texture_blocks(name)],
                             log_filename)


def store_all_feature_vectors(method=0, log_filename=None):
    forms_filenames = [n for n in os.listdir(FLAGS.DEFAULT_DATASET_PATH)]
    store_feature_vectors_of_list(forms_filenames, method, log_filename=log_filename)


def get_form_feature_vectors(form_filename, method='LBP'):
    if form_filename in features_dict[method]:
        return features_dict[method][form_filename]
    else:
        return [getFeatureVector(b, method) for b in pre.get_texture_blocks(form_filename)]


def getFeatureVector(textureBlock, method=0):
    """
    method: integer
    0: LBP
    1: CSLBCoP

    return: numpy vector
    """

    if method == 'LBP' or method == 0:
        I_LBP = np.uint8(local_binary_pattern(textureBlock, 8, 1))
        # n_bins = I_LBP.max() + 1
        hist, bin_edges = np.histogram(I_LBP.ravel(), np.r_[0:256])  # ALERT
        vector = hist.reshape(1, -1)

    elif method == 'CSLBCoP' or method == 1:
        vector = get_CSLBCoP_vector(textureBlock)

    elif method == 'LPQ' or method == 2:
        vector = LPQ(textureBlock, 3)

    else:
        raise AttributeError("Method number not found in feature extraction")

    return vector


def LPQ(textureBlock, winSize=3):
    # textureBlock is expected to be grayscale
    # winSize is expected to be odd and has min value of 3
    STFTalpha = 1 / winSize
    sigmaS = (winSize - 1) / 4
    sigmaA = 8 / (winSize - 1)

    img = textureBlock.astype(np.float)
    radius = (winSize - 1) / 2
    x = np.arange(-radius, radius + 1)  # Form spatial coordinates in window
    r = np.arange(1,
                  radius + 1)  # Form coordinates of positive half of the Frequency domain (Needed for Gaussian derivative)

    n = len(x)
    f = 1.0
    rho = 0.95
    [xp, yp] = np.meshgrid(np.arange(1, (n + 1)), np.arange(1, (n + 1)))
    pp = np.concatenate((xp, yp)).reshape(2, -1)
    dd = euc_dist(pp.T)  # squareform(pdist(...)) would do the job, too...
    C = np.power(rho, dd)

    w0 = (x * 0.0 + 1.0)
    w1 = np.exp(-2 * np.pi * 1j * x * f / n)
    w2 = np.conj(w1)

    q1 = w0.reshape(-1, 1) * w1
    q2 = w1.reshape(-1, 1) * w0
    q3 = w1.reshape(-1, 1) * w1
    q4 = w1.reshape(-1, 1) * w2

    u1 = np.real(q1)
    u2 = np.imag(q1)
    u3 = np.real(q2)
    u4 = np.imag(q2)
    u5 = np.real(q3)
    u6 = np.imag(q3)
    u7 = np.real(q4)
    u8 = np.imag(q4)

    M = np.matrix(
        [u1.flatten(1), u2.flatten(1), u3.flatten(1), u4.flatten(1), u5.flatten(1), u6.flatten(1), u7.flatten(1),
         u8.flatten(1)])

    D = np.dot(np.dot(M, C), M.T)
    U, S, V = np.linalg.svd(D)

    Qa = convolve2d(convolve2d(textureBlock, w0.reshape(-1, 1), mode='same'), w1.reshape(1, -1), mode='same')
    Qb = convolve2d(convolve2d(textureBlock, w1.reshape(-1, 1), mode='same'), w0.reshape(1, -1), mode='same')
    Qc = convolve2d(convolve2d(textureBlock, w1.reshape(-1, 1), mode='same'), w1.reshape(1, -1), mode='same')
    Qd = convolve2d(convolve2d(textureBlock, w1.reshape(-1, 1), mode='same'), w2.reshape(1, -1), mode='same')

    Fa = np.real(Qa)
    Ga = np.imag(Qa)
    Fb = np.real(Qb)
    Gb = np.imag(Qb)
    Fc = np.real(Qc)
    Gc = np.imag(Qc)
    Fd = np.real(Qd)
    Gd = np.imag(Qd)

    F = np.array(
        [Fa.flatten(1), Ga.flatten(1), Fb.flatten(1), Gb.flatten(1), Fc.flatten(1), Gc.flatten(1), Fd.flatten(1),
         Gd.flatten(1)])
    G = np.dot(V.T, F)

    t = 0

    # Calculate the LPQ Patterns:
    B = (G[0, :] >= t) * 1 + (G[1, :] >= t) * 2 + (G[2, :] >= t) * 4 + (G[3, :] >= t) * 8 + (G[4, :] >= t) * 16 + (
            G[5, :] >= t) * 32 + (G[6, :] >= t) * 64 + (G[7, :] >= t) * 128
    B = np.reshape(B, np.shape(Fa))

    # And finally build the histogram:
    h, b = np.histogram(B, bins=256, range=(0, 255))

    return h.reshape(1, -1)  # return rows


def euc_dist(X):
    Y = X = X.astype(np.float)
    XX = np.sum(X * X, axis=1)[:, np.newaxis]
    YY = XX.T
    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, distances)
    distances.flat[::distances.shape[0] + 1] = 0.0
    return np.sqrt(distances)


def get_CSLBP(block, radius=1):
    new_image = np.zeros((len(block), len(block[0])))
    padded_block = np.zeros((len(block) + 2, len(block[0]) + 2))
    padded_block[1:len(padded_block) - 1, 1:len(padded_block[0]) - 1] = block

    for x in range(len(block)):
        for y in range(len(block[x])):
            i = x + 1
            j = y + 1
            roi = padded_block

            p = np.array([roi[i, j + 1], roi[i + 1, j + 1], roi[i + 1, j], roi[i + 1, j - 1]])
            n = np.array([roi[i, j - 1], roi[i - 1, j - 1], roi[i - 1, j], roi[i - 1, j + 1]])

            T = 3.0

            diff = p - n
            diff[diff <= T] = 0.0
            diff[diff > T] = 1.0

            new_image[x, y] = np.sum(np.array([v * np.power(2, a) for a, v in enumerate(diff)]))

    return new_image


def get_CSLBCoP_vector(img_block):
    cslbp_map = get_CSLBP(img_block)
    angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    res = greycomatrix(cslbp_map.astype(int), [1], angles, levels=16)  # ALERT
    return np.array(res).reshape((1, -1))


if __name__ == '__main__':
    print("Basic test:")
    test_img = io.imread(
        r"E:\CUFE CHS\Semester 9 (Senior 2)\Pattern Recognition\Project\Writer_Identification\test_CSLBCoP.PNG")
    test_img = rgb2gray(test_img)
    feature = get_CSLBCoP_vector(test_img)
    print("Feature vector:", feature)
