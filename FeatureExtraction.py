from skimage.feature import local_binary_pattern
import numpy as np

def getFeatureVector(textureBlock, n_bins):
	I_LBP = np.uint8(local_binary_pattern(textureBlock, 8, 1))
	#n_bins = I_LBP.max() + 1
	hist, bin_edges = np.histogram(I_LBP.ravel(), np.r_[0:n_bins])
	return hist