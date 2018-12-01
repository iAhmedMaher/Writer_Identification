from skimage.feature import local_binary_pattern
from scipy.signal import convolve2d
import numpy as np

def getFeatureVector(textureBlock, n_bins):
	I_LBP = np.uint8(local_binary_pattern(textureBlock, 8, 1))
	#n_bins = I_LBP.max() + 1
	hist, bin_edges = np.histogram(I_LBP.ravel(), np.r_[0:n_bins])
	return hist

def LPQ (textureBlock, winSize = 3):
	# textureBlock is expected to be grayscale
	# winSize is expected to be odd and has min value of 3
	STFTalpha = 1 / winSize
	sigmaS = (winSize - 1) / 4
	sigmaA = 8 / (winSize - 1)

	img = float(textureBlock)
	radius = (winSize-1)/2
	x = np.arange(-radius, radius+1)  # Form spatial coordinates in window
	r = np.arange(1, radius + 1)  # Form coordinates of positive half of the Frequency domain (Needed for Gaussian derivative)

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

	M = np.matrix([u1.flatten(1), u2.flatten(1), u3.flatten(1), u4.flatten(1), u5.flatten(1), u6.flatten(1), u7.flatten(1), u8.flatten(1)])

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

	F = np.array([Fa.flatten(1), Ga.flatten(1), Fb.flatten(1), Gb.flatten(1), Fc.flatten(1), Gc.flatten(1), Fd.flatten(1), Gd.flatten(1)])
	G = np.dot(V.T, F)

	t = 0

	# Calculate the LPQ Patterns:
	B = (G[0, :] >= t) * 1 + (G[1, :] >= t) * 2 + (G[2, :] >= t) * 4 + (G[3, :] >= t) * 8 + (G[4, :] >= t) * 16 + (
				G[5, :] >= t) * 32 + (G[6, :] >= t) * 64 + (G[7, :] >= t) * 128
	B = np.reshape(B, np.shape(Fa))

	# And finally build the histogram:
	h, b = np.histogram(B, bins=256, range=(0, 255))

	return h


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

