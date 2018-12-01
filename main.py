from Preprocessing import *
from Classification import *
from FeatureExtraction import *
import random

import skimage.io as io
from os import listdir
from os.path import isfile, join

testWriterFileName=None
X_Train=[]
X_Test=[]
Y_Train=[]
Y_Test=[]

def processTestImages(dataPath):
	global testWriterFileName
	X=[]
	Y=[]
	test_img = io.imread(join(dataPath, testWriterFileName))
	textureBlocks = Preprocessing(test_img)
	# for textureBlock in textureBlocks:
	X.append(getFeatureVector(textureBlocks, 256))
	Y.append(str(testWriterFileName).split('_')[0])
	return X,Y

def processTrainingImages(dataPath, numWriters):
	global testWriterFileName
	# Removed check if isfile(join(trainingDataPath, f))
	fileNames = listdir(dataPath)
	availableWriters=list(range(numWriters))
	rndmWriters = []
	while(len(rndmWriters)<3):
		rndmWriter = random.randint(0, len(availableWriters))
		#Chosen writer must have at least 2 forms
		writerForms = [f for f in fileNames if f.startswith(availableWriters[rndmWriter])]
		if (len(writerForms) <= 1 and len(rndmWriters)<2) or (len(writerForms)<=2 and len(rndmWriters)==2):
			availableWriters.pop(rndmWriter)
			continue
		rndmWriters.append(availableWriters[rndmWriter])
		availableWriters.pop(rndmWriter)

	X=[]
	Y=[]
	for i in range(3):
		writerForms=[f for f in fileNames if f.startswith(i)]
		availableForms = list(range(writerForms))

		#Process first writer form
		for i in range(2):
			rndmForm = random.randint(0,len(availableForms))
			form = writerForms[availableForms[rndmForm]]
			training_img = io.imread(join(dataPath, form))
			textureBlocks = Preprocessing(training_img)
			#for textureBlock in textureBlocks:
			X.append(getFeatureVector(textureBlocks, 256))
			Y.append(i)
			availableForms.pop()

		if i==2:
			rndmForm = random.randint(0, len(availableForms))
			testWriterFileName = writerForms[availableForms[rndmForm]]

	return X,Y

"""Main Function"""
if __name__ == '__main__':
	trainingDataPath = "Training"
	testDataPath = "Test"

	X_Train,Y_Train = processImages(trainingDataPath)
	X_Test, Y_Test = processImages(testDataPath)

	trainingFiles = [f for f in listdir(trainingDataPath)]
	adaboost_clf(Y_Train, X_Train, numClassifiers=50, learnRate=1.5, clfNum=3)
	testFiles = [f for f in listdir(testDataPath)]
	predict_clf(Y_Test,X_Test)



