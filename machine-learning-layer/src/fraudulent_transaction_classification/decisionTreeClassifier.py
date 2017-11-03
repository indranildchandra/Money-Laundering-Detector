import numpy as np
from numpy import genfromtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.preprocessing import LabelEncoder
import pandas
import csv

from os import walk

import pickle

data_path = "./../../datasets/dataset_primary_segmented.csv"

dataMat = pandas.read_csv(data_path, sep=",",header=0)
data = dataMat.as_matrix()

def writeArrayToCsv(arr,file):
	with open(file, "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(arr)

	#this is to remove the newlines
	with open(file, "r") as f:
		lines = f.readlines()
		lines = [line for i,line in enumerate(lines) if i%2==0]

	with open(file,"w") as f:
		header = "step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,accountType,isFraud,isFlaggedFraud"
		f.write(header + "\n" + "".join(lines))

##uncomment for saving the segmented data into respective files
# segData = {}
# for i in range(data.shape[0]):
# 	segment = "segment_" + str(data[i,0])
# 	if segData.get(segment,-1) == -1:
# 		segData[segment] = []
# 	segData[segment].append(data[i,1:])

# for segment, data in segData.items():
# 	file = "./../../datasets/segments/"+segment+".csv"
# 	writeArrayToCsv(data,file)

for (dirpath, dirnames, filenames) in walk("./../../datasets/segments"):
	for file in filenames:
		data_path = "./../../datasets/segments/" + file
		print(data_path)
		dfX = pandas.read_csv(data_path, sep=",",header=0)
		data = dfX.as_matrix()

		X = data[:,:-2]
		y = data[:,-2]
		if 1 not in y:
			continue

		#remove null values from categorical columns specifically
		def removeNulls(data, col):
			for i in range(data.shape[0]):
				if type(data[i,col]) is float and np.isnan(data[i,col]):
					data[i,col] = 'NA'

		cat_cols = [1,3,5,7]
		# removeNulls(X,5)
		#change the original categorical data to numbers for input to model
		for i in cat_cols:
			le = LabelEncoder()
			le = le.fit(X[:,i])
			X[:,i] = le.transform(X[:,i])

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
		while 1 not in y_train:
			print("in while")
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

		model = pickle.load(open("./../../models/fraud_classifier_tree/"+file.split(".")[0]+".dat", "rb"))
		
		##uncomment to train
		# model = tree.DecisionTreeClassifier()
		# model.fit(X_train,y_train.astype(int))

		# clf.fit(X_train[:3,:], [1,0,0])

		# pickle.dump(model, open("./../../models/fraud_classifier_tree/"+file.split(".")[0]+".dat", "wb"))
		# print("models saved in models folder")

		preds = model.predict(X_test)		
		print(y_test,preds)
		f1 = f1_score(y_test.astype(int), preds.astype(int))
		print("f1 score")
		print(f1)


		tested = np.concatenate((X_test,np.expand_dims(y_test,axis=1),np.expand_dims(preds,axis=1)), axis=1)
		with open("./../../testCases/testTreeClassifier_"+file.split(".")[0]+".csv", "w") as f:
			writer = csv.writer(f)
			writer.writerows(tested)

			#this is to remove the newlines
			with open("./../../testCases/testTreeClassifier_"+file.split(".")[0]+".csv", "r") as f:
				lines = f.readlines()
				lines = [line for i,line in enumerate(lines) if i%2==0]

			with open("./../../testCases/testTreeClassifier_"+file.split(".")[0]+".csv","w") as f:
				header = "step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,accountType,isFraud,isFlaggedFraud"
				f.write(header + "\n" + "".join(lines))