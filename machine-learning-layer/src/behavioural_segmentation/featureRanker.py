from sklearn import tree
import numpy as np
import math

from numpy import genfromtxt

import pandas
import pickle
import time
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

#read training data
dataframeX = pandas.read_csv('./../../datasets/dataset_secondary.csv', sep=",",header=0)
col_names = list(dataframeX.columns.values)

#convert to matrix for input
dataMat = dataframeX.as_matrix()
X = dataMat[:,1:-2]
Y = dataMat[:,-1]

assert (X.shape[0] == Y.shape[0]), "No of products and classes not equal"

#remove empty fields from categorical columns
def removeNulls(X, col):
	for i in range(X.shape[0]):
		if type(X[i,col]) is float and np.isnan(X[i,col]):
			X[i,col] = 'NA'

#train the model
model = tree.DecisionTreeClassifier()
model.fit(X,Y.astype(int))

#save the model
pickle.dump(model, open("./../../models/tree_classifier_model.dat", "wb"))
print("tree_classifier_model.dat saved in models folder")

feature_imp = model.feature_importances_

sorted_feature_vals = np.sort(feature_imp)
sorted_feature_indexes = np.argsort(feature_imp)

print("Significant Features in decreasing order of importance: ")
for i in reversed(sorted_feature_indexes):
	print(col_names[i+2].replace('\t',''), " \t->\t ",feature_imp[i])


########################################################################
#output
########################################################################
# Significant Features in decreasing order of importance:
# balance_difference_60   		  ->        0.297249822847
# balance_difference_90   		  ->        0.292937784263
# outgoing_domestic_amount_60     ->        0.112128004209
# incoming_foreign_amount_30      ->        0.0809464605337
# outgoing_domestic_amount_90     ->        0.0803045191632
# outgoing_foreign_amount_90      ->        0.0328515844364
# incoming_domestic_count_30      ->        0.0323738450738
# incoming_domestic_count_60      ->        0.0188196650244
# outgoing_foreign_amount_60      ->        0.0170659207114
# incoming_foreign_amount_60      ->        0.0163670574818
# incoming_domestic_count_90      ->        0.00795390742756
# incoming_foreign_amount_90      ->        0.00761129603537
# outgoing_foreign_amount_30      ->        0.00339013279358
# incoming_domestic_amount_90     ->        0.0
# outgoing_domestic_amount_30     ->        0.0
# balance_difference_30 		  ->        0.0
# outgoing_foreign_count_90       ->        0.0
# outgoing_foreign_count_60       ->        0.0
# incoming_foreign_count_30       ->        0.0
# outgoing_domestic_count_90      ->        0.0
# outgoing_foreign_count_30       ->        0.0
# incoming_foreign_count_90       ->        0.0
# incoming_foreign_count_60       ->        0.0
# outgoing_domestic_count_30      ->        0.0
# outgoing_domestic_count_60      ->        0.0
# incoming_domestic_amount_60     ->        0.0