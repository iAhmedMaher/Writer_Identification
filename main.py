from Preprocessing import *
from Classification import *
from FeatureExtraction import *

import skimage.io as io
from os import listdir
from os.path import isfile, join

X_Train=[]
X_Test=[]
Y_Train=[]
Y_Test=[]

def processImages(dataPath):
	# Removed check if isfile(join(trainingDataPath, f))
	fileNames = [f for f in listdir(dataPath)]
	X=[]
	Y=[]
	for i in range(len(fileNames)):
		training_img = io.imread(join(dataPath, fileNames[i]))
		textureBlocks = Preprocessing(training_img)
		#for textureBlock in textureBlocks:
		X.append(getFeatureVector(textureBlocks))
		Y.append(fileNames[i].split('_')[0])
	return X,Y


"""Main Function"""
if __name__ == '__main__':
	trainingDataPath = "Training"
	testDataPath = "Test"

	X_Train,Y_Train = processImages(trainingDataPath)
	X_Test, Y_Test = processImages(testDataPath)

	trainingFiles = [f for f in listdir(trainingDataPath)]
	adaboost_clf(Y_Train, X_Train, numClassifiers=50, learnRate=1.5, clfNum=2)
	testFiles = [f for f in listdir(testDataPath)]
	predict_clf(Y_Test,X_Test)



