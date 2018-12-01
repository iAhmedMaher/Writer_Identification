from skimage.feature import local_binary_pattern
import numpy as np
from skimage.feature import greycomatrix
import skimage.io as io
from skimage.color import rgb2gray

<<<<<<< HEAD
def getFeatureVector(textureBlock, n_bins):
	I_LBP = np.uint8(local_binary_pattern(textureBlock, P=8, R=3))
	#n_bins = I_LBP.max() + 1
	hist, bin_edges = np.histogram(I_LBP.ravel(), np.r_[0:n_bins])
	return hist
=======

def getFeatureVector(textureBlock, method=1):
    """
    method: integer
    0: LBP
    1: CSLBCoP

    return: numpy vector
    """

    if method == 0:
        I_LBP = np.uint8(local_binary_pattern(textureBlock, 8, 1))
        # n_bins = I_LBP.max() + 1
        hist, bin_edges = np.histogram(I_LBP.ravel(), np.r_[0:256])  # ALERT
        vector = hist

    elif method == 1:
        vector = get_CSLBCoP_vector(textureBlock)

    else:
        raise AttributeError()

    return vector


def get_CSLBP(block, radius=1):
    new_image = np.zeros((len(block), len(block[0])))
    padded_block = np.zeros((len(block)+2, len(block[0])+2))
    padded_block[1:len(padded_block)-1, 1:len(padded_block[0])-1] = block

    for x in range(len(block)):
        for y in range(len(block[x])):
            i = x+1
            j = y+1
            roi = padded_block

            p = np.array([roi[i, j+1], roi[i+1, j+1], roi[i+1, j], roi[i+1, j-1]])
            n = np.array([roi[i, j-1], roi[i-1, j-1], roi[i-1, j], roi[i-1, j+1]])

            T = 3.0

            diff = p-n
            diff[diff <= T] = 0.0
            diff[diff > T] = 1.0

            new_image[x, y] = np.sum(np.array([v * np.power(2, a) for a, v in enumerate(diff)]))

    return new_image


def get_CSLBCoP_vector(img_block):

    cslbp_map = get_CSLBP(img_block)
    angles = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    res = greycomatrix(cslbp_map.astype(int), [1], angles, levels=16)  # ALERT
    return np.array(res).reshape((-1, 1))


if __name__ == '__main__':

    print("Basic test:")
    test_img = io.imread(r"D:\Projects\Pattern_Recognition\Writer_Identification\test_CSLBCoP.PNG")
    test_img = rgb2gray(test_img)
    feature = get_CSLBCoP_vector(test_img)
    print("Feature vector:", feature)
>>>>>>> ce866a33307d2b8f88c7b05ffe310c32104d6e17
