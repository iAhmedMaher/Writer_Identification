#from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# The trained classifier to be used for prediction
clf=None

"""Predicting Writer
Y_test: The target values for the test set. - 
X_test: The input features for the test set.
"""
def predict_clf(Y_test, X_test):
	global clf
	prediction = clf.predict(X_test)
	print(Y_test)
	print(prediction)
	accuracy = sum(Y_test==prediction)/len(Y_test)
	print("This paper belongs to class "+str(prediction))
	print("The classification accuracy: "+str(accuracy))
	if(accuracy<1):
		exit()
	return

"""AdaBoost Implementation
Y_train: The target values for the training set. - 
X_train: The input features for the training set. - 
numClassifiers: The number of iterations of the AdaBoost Algorithm. - 
clfNum: The classifier to be used. (1=Decision Tree/2=KNN)
"""
def adaboost_clf(Y_train, X_train, numClassifiers, learnRate, clfNum):
	global clf

	classifiers = {
		1: DecisionTreeClassifier(max_depth=1, random_state=1),
		2: KNeighborsClassifier(n_neighbors=3), #Can add weights='distance',
		3: SVC(C=1.0,kernel='linear')
	}
	clf = classifiers.get(clfNum,None)

	if (clf is None):
		print("Classifier number not recognized!")
		return

	if(clfNum==1):
		clf = AdaBoostClassifier(clf,
		    n_estimators=numClassifiers,
		    learning_rate=learnRate,
		    algorithm="SAMME")

	clf.fit(X_train, Y_train)
	return

	'''
	bdt_real = AdaBoostClassifier(
	    #DecisionTreeClassifier(max_depth=2),
		clf,
	    n_estimators=numClassifiers,
	    learning_rate=1)

	bdt_discrete = AdaBoostClassifier(
	    #DecisionTreeClassifier(max_depth=2),
		clf,
	    n_estimators=numClassifiers,
	    learning_rate=1.5,
	    algorithm="SAMME")

	bdt_real.fit(X_train, Y_train)
	bdt_discrete.fit(X_train, Y_train)
	'''

	'''
	real_test_errors = []
	discrete_test_errors = []

	for real_test_predict, discrete_train_predict in zip(
	        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
	    real_test_errors.append(
	        1. - accuracy_score(real_test_predict, Y_test))
	    discrete_test_errors.append(
	        1. - accuracy_score(discrete_train_predict, Y_test))

	n_trees_discrete = len(bdt_discrete)
	n_trees_real = len(bdt_real)

	# Boosting might terminate early, but the following arrays are always
	# n_estimators long. We crop them to the actual number of trees here:
	discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
	real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
	discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

	plt.figure(figsize=(15, 5))

	plt.subplot(131)
	plt.plot(range(1, n_trees_discrete + 1),
	         discrete_test_errors, c='black', label='SAMME')
	plt.plot(range(1, n_trees_real + 1),
	         real_test_errors, c='black',
	         linestyle='dashed', label='SAMME.R')
	plt.legend()
	plt.ylim(0.18, 0.62)
	plt.ylabel('Test Error')
	plt.xlabel('Number of Trees')

	plt.subplot(132)
	plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
	         "b", label='SAMME', alpha=.5)
	plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
	         "r", label='SAMME.R', alpha=.5)
	plt.legend()
	plt.ylabel('Error')
	plt.xlabel('Number of Trees')
	plt.ylim((.2,
	         max(real_estimator_errors.max(),
	             discrete_estimator_errors.max()) * 1.2))
	plt.xlim((-20, len(bdt_discrete) + 20))

	plt.subplot(133)
	plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
	         "b", label='SAMME')
	plt.legend()
	plt.ylabel('Weight')
	plt.xlabel('Number of Trees')
	plt.ylim((0, discrete_estimator_weights.max() * 1.2))
	plt.xlim((-20, n_trees_discrete + 20))

	# prevent overlapping y-axis labels
	plt.subplots_adjust(wspace=0.25)
	plt.show()'''
