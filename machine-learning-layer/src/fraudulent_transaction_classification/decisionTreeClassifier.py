import numpy as np
from numpy import genfromtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import metrics
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


data_path = config.segmentGenerator.train_full_path
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

cat_cols = [0,4]
#TODO: during actual inference le will have to be precomputed and stores somewhere
le_arr = []
for i in cat_cols:
	le_arr.append(LabelEncoder())
	le_arr[-1] = le_arr[-1].fit(X[:,i])

# removeNulls(X,5)
def removeLabels(X):
	#change the original categorical data to numbers for input to model
	for index,i in enumerate(cat_cols):
		X[:,i] = le_arr[index].transform(X[:,i])

	return X

y_true = []
y_pred = []
for file in os.listdir(config.decisionTree.data_train_test_path):
	if os.path.isdir(os.path.join(config.decisionTree.data_train_test_path,file)):
			continue
	if config.train:
		data_path = os.path.join(config.decisionTree.train_path,file)
		# print(data_path)
		if os.path.exists(data_path):
			print('Training {}'.format(file))
			dfX = pandas.read_csv(data_path, sep=",",header=0)
			data_train = dfX.as_matrix()

			X_train = removeLabels(data_train[:,:-1])
			y_train = data_train[:,-1]

			model = tree.DecisionTreeClassifier()
			model.fit(X_train,y_train.astype(int))


			pickle.dump(model, open(os.path.join(config.decisionTree.model_path,file.split(".")[0]+".dat"), "wb"))
			print("models saved in models folder")
	else:
		data_path = os.path.join(config.decisionTree.test_path,file)
		print("Testing {}".format(data_path))
		dfX = pandas.read_csv(data_path, sep=",",header=0)
		data_test = dfX.as_matrix()

		X_test = removeLabels(data_test[:,:-1])
		y_test = data_test[:,-1]


		if os.path.exists(os.path.join(config.decisionTree.model_path,file.split(".")[0]+".dat")):
			model = pickle.load(open(os.path.join(config.decisionTree.model_path,file.split(".")[0]+".dat"), "rb"))

			preds = model.predict(X_test)		

		else:
			preds = np.zeros(y.shape)	

		if len(y_pred)>0:
			y_pred = np.concatenate((y_pred,preds),axis=0)
			y_true = np.concatenate((y_true,y_test),axis=0)
		else:
			y_pred = preds
			y_true = y_test

		tested = np.concatenate((X_test,np.expand_dims(y_test,axis=1),np.expand_dims(preds,axis=1)), axis=1)
		with open(config.decisionTree.test_path_prefix+file.split(".")[0]+".csv", "w") as f:
			writer = csv.writer(f)
			writer.writerows(tested)

			#this is to remove the newlines
			with open(config.decisionTree.test_path_prefix+file.split(".")[0]+".csv", "r") as f:
				lines = f.readlines()
				lines = [line for i,line in enumerate(lines)]

			with open(config.decisionTree.test_path_prefix+file.split(".")[0]+".csv","w") as f:
				header = """trans_type,amount,oldbalanceOrg,\
oldbalanceDest,accountType,\
incoming_domestic_amount_30_src,incoming_domestic_amount_60_src,incoming_domestic_amount_90_src,\
outgoing_domestic_amount_30_src,outgoing_domestic_amount_60_src,outgoing_domestic_amount_90_src,\
incoming_foreign_amount_30_src,incoming_foreign_amount_60_src,incoming_foreign_amount_90_src,\
outgoing_foreign_amount_30_src,outgoing_foreign_amount_60_src,outgoing_foreign_amount_90_src,\
incoming_domestic_count_30_src,incoming_domestic_count_60_src,incoming_domestic_count_90_src,\
outgoing_domestic_count_30_src,outgoing_domestic_count_60_src,outgoing_domestic_count_90_src,\
incoming_foreign_count_30_src,incoming_foreign_count_60_src,incoming_foreign_count_90_src,\
outgoing_foreign_count_30_src,outgoing_foreign_count_60_src,outgoing_foreign_count_90_src,\
balance_difference_30_src,balance_difference_60_src,balance_difference_90_src,\
incoming_domestic_amount_30_dst,incoming_domestic_amount_60_dst,incoming_domestic_amount_90_dst,\
outgoing_domestic_amount_30_dst,outgoing_domestic_amount_60_dst,outgoing_domestic_amount_90_dst,\
incoming_foreign_amount_30_dst,incoming_foreign_amount_60_dst,incoming_foreign_amount_90_dst,\
outgoing_foreign_amount_30_dst,outgoing_foreign_amount_60_dst,outgoing_foreign_amount_90_dst,\
incoming_domestic_count_30_dst,incoming_domestic_count_60_dst,incoming_domestic_count_90_dst,\
outgoing_domestic_count_30_dst,outgoing_domestic_count_60_dst,outgoing_domestic_count_90_dst,\
incoming_foreign_count_30_dst,incoming_foreign_count_60_dst,incoming_foreign_count_90_dst,\
outgoing_foreign_count_30_dst,outgoing_foreign_count_60_dst,outgoing_foreign_count_90_dst,\
balance_difference_30_dst,balance_difference_60_dst,balance_difference_90,\
isFraud,isFlaggedFraud"""
				f.write(header + "\n" + "".join(lines))


f1 = f1_score(y_true.astype(int), y_pred.astype(int))
print("f1 score")
print(f1)

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
auc_val = metrics.auc(fpr, tpr)
print('auc_val')
print(auc_val)