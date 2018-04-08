import numpy as np
from numpy import genfromtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
import pandas
import csv

from os import walk
import os

import pickle

import sys
sys.path.append('../..')
import config

data_path = config.segmentGenerator.out_path

dataMat = pandas.read_csv(data_path, sep=",",header=0)
data = dataMat.as_matrix()

def writeArrayToCsv(arr,file):
	with open(file, "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(arr)

	#this is to remove the newlines
	with open(file, "r") as f:
		lines = f.readlines()
		lines = [line for i,line in enumerate(lines)]

	with open(file,"w") as f:
		header = "step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,accountType,isFraud,isFlaggedFraud"
		f.write(header + "\n" + "".join(lines))

segData = {}
for i in range(data.shape[0]):
	segment = "segment_" + str(data[i,0])
	if segData.get(segment,-1) == -1:
		segData[segment] = []
	segData[segment].append(data[i,1:])

for segment, data in segData.items():
	file = os.path.join(config.segmentGenerator.segment_path,segment+".csv")
	writeArrayToCsv(data,file)

for file in os.listdir(config.segmentGenerator.segment_path):
	for filenum, file in enumerate(filenames):
		data_path = os.path.join(config.segmentGenerator.segment_path,file)
		print(data_path)
		dfX = pandas.read_csv(data_path, sep=",",header=0)
		data = dfX.as_matrix()

		X = data[:,:-2]
		y = data[:,-2]


		#remove null values from categorical columns specifically
		def removeNulls(data, col):
			for i in range(data.shape[0]):
				if type(data[i,col]) is float and np.isnan(data[i,col]):
					data[i,col] = 'NA'

		cat_cols = [1,3,5,7]
		# removeNulls(X,5)
		#TODO: during actual inference le will have to be precomputed and stores somewhere
		le = LabelEncoder()
		le = le.fit(X[:,i])

		# removeNulls(X,5)
		def removeLabels(X):
			#change the original categorical data to numbers for input to model
			for i in cat_cols:
				X[:,i] = le.transform(X[:,i])


		if config.train:
			if 1 not in y:
				continue

			data_path = os.path.join(config.segmentGenerator.train_segment_path,file)
			# print(data_path)
			dfX = pandas.read_csv(data_path, sep=",",header=0)
			data_train = dfX.as_matrix()

			X = removeLabels(data_train[:-1])
			y = data_train[-1]

			clf = SVC()
			clf.fit(X, np.ndarray.flatten(y.astype(int))) #uncomment for training

			pickle.dump(clf, open(os.path.join(config.svmClassifier.model_path,file.split(".")[0]+".dat"), "wb"))
			print("models saved in models folder")
		else:
			data_path = os.path.join(config.segmentGenerator.test_segment_path,file)
			# print(data_path)
			dfX = pandas.read_csv(data_path, sep=",",header=0)
			data_test = dfX.as_matrix()

			X = removeLabels(data_test[:-1])
			y = data_test[-1]

			if os.path.exists(os.path.join(config.svmClassifier.model_path,file.split(".")[0]+".dat")):
				clf = pickle.load(open(os.path.join(config.svmClassifier.model_path,file.split(".")[0]+".dat"), "rb"))

				preds = clf.predict(X)		
				print(y,preds)
				f1 = f1_score(y.astype(int), preds.astype(int))
				print("f1 score")
				print(f1)
			else:
				preds = np.zeros(y.shape)	
				print(y,preds)
				f1 = f1_score(y.astype(int), preds.astype(int))
				print("f1 score")
				print(f1)

			tested = np.concatenate((X,np.expand_dims(y,axis=1),np.expand_dims(preds,axis=1)), axis=1)
			with open(config.svmClassifier.test_path_prefix+file.split(".")[0]+".csv", "w") as f:
				writer = csv.writer(f)
				writer.writerows(tested)

				#this is to remove the newlines
				with open(config.svmClassifier.test_path_prefix+file.split(".")[0]+".csv", "r") as f:
					lines = f.readlines()
					lines = [line for i,line in enumerate(lines)]

				with open(config.svmClassifier.test_path_prefix+file.split(".")[0]+".csv","w") as f:
					header = "step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,accountType,isFraud,isFlaggedFraud"
					f.write(header + "\n" + "".join(lines))