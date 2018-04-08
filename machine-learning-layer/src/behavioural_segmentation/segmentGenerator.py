import time

import json

import pandas
import numpy as np
import csv

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from os import walk
import os

import sys
sys.path.append('../..')
import config

def getClusterPredictions(data, true_k):
	model = KMeans(n_clusters=true_k)
	model.fit(data)
	prediction = model.predict(data)

	return prediction


def getBestCluster(X,_min=2,_max=10):
	selected_cluster = 0
	previous_sil_coeff = 0.001 #some random small number not 0
	sc_vals = []
	for n_cluster in range(_min, _max):
	    kmeans = KMeans(n_clusters=n_cluster).fit(X)
	    label = kmeans.labels_

	    sil_coeff = silhouette_score(X, label, metric='euclidean', sample_size=1000)
	    sc_vals.append(sil_coeff)
	    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

	    percent_change = (sil_coeff-previous_sil_coeff)*100/previous_sil_coeff

	    # return when below a threshold of 1%
	    if percent_change<1:
	    	selected_cluster = n_cluster-1

	    previous_sil_coeff = sil_coeff

	return selected_cluster or _max, sc_vals

data_path = config.secondaryDataGenerator.out_path

print("Reading dataset...")
X_dataframe = pandas.read_csv(data_path, sep=",",header=0)
X = X_dataframe.as_matrix()
print("Read dataset")
col_names = list(X_dataframe.columns.values)


X_trimmed_features = np.zeros((X.shape[0],1))
features_to_select = ["balance_difference_60","balance_difference_90", "outgoing_domestic_amount_60","incoming_foreign_amount_30", "outgoing_domestic_amount_90","outgoing_foreign_amount_90", "incoming_domestic_count_30", "incoming_domestic_count_60", "outgoing_foreign_amount_60", "incoming_foreign_amount_60", "incoming_domestic_count_90", "incoming_foreign_amount_90", "outgoing_foreign_amount_30" ]
for feature in features_to_select:
	X_trimmed_features = np.concatenate((X_trimmed_features,np.expand_dims(X_dataframe[feature],axis=1)),axis=1)
X_trimmed_features = X_trimmed_features[:,1:]

if config.train:
	true_k, sc_vals = getBestCluster(X_trimmed_features,_min=2,_max=10)
else:
	#TODO: write to config file the number of clusters after training
	true_k = 7


print("Best Cluster")
print(true_k)

# plt.plot(range(len(sc_vals)), sc_vals)
# plt.show()

prediction = getClusterPredictions(X_trimmed_features, true_k)

#form segmentation dict
seg_dict = {}
for i in range(X.shape[0]):
	seg_dict[X[i,0]] = prediction[i]


#read the primary data
data_path_primary = config.secondaryDataGenerator.out_path_primary

print("Reading dataset...")
X_dataframe_pri = pandas.read_csv(data_path_primary, sep=",",header=0)
X_pri = X_dataframe_pri.as_matrix()
print("Read dataset")
col_names = list(X_dataframe_pri.columns.values)

X_with_segments = []
with open(config.segmentGenerator.out_path,"w") as f:
	# X_with_segments = np.concatenate((np.expand_dims(prediction,axis=1),X_pri),axis=1)
	for i in range(X_pri.shape[0]):
		X_with_segments.append(np.concatenate(([[seg_dict[X_pri[i,3]]]],np.expand_dims(X_pri[i,:],axis=0)),axis=1)[0])
	writer = csv.writer(f)
	writer.writerows(X_with_segments)

#this is to remove the newlines
with open(config.segmentGenerator.out_path, "r") as f:
	lines = f.readlines()
	lines = [line for i,line in enumerate(lines) if i%2==0]

with open(config.segmentGenerator.out_path,"w") as f:
	header = """segment,step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,\
	oldbalanceDest,accountType,isFraud,isFlaggedFraud,\
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
balance_difference_30_dst,balance_difference_60_dst,balance_difference_90"""
	f.write(header + "\n" + "".join(lines))

print("Saved segmented dataset")




#save the segmented data into respective files
def writeArrayToCsv(arr,file,split=True):
	print('Writing into file {}'.format(file))
	with open(file, "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(arr)

	#this is to remove the newlines
	with open(file, "r") as f:
		lines = f.readlines()
		lines = [line for i,line in enumerate(lines)]

	with open(file,"w") as f:
		if not split:
			header = """step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,\
	oldbalanceDest,accountType,isFraud,isFlaggedFraud,\
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
balance_difference_30_dst,balance_difference_60_dst,balance_difference_90"""
		else:
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
isFraud"""		
		f.write(header + "\n" + "".join(lines))

segData = {}
X_with_segments = np.array(X_with_segments)
for i in range(X_with_segments.shape[0]):
	segment = "segment_" + str(X_with_segments[i,0])
	if segData.get(segment,-1) == -1:
		segData[segment] = []
	segData[segment].append(X_with_segments[i,1:])

for segment, X_with_segments in segData.items():
	file = os.path.join(config.segmentGenerator.segment_path,segment+".csv")
	writeArrayToCsv(X_with_segments,file,split=False)


train_full = []
test_full = []
fraud_col = 8
src_name_col = 3
dst_name_col = 5
for (dirpath, dirnames, filenames) in walk(config.segmentGenerator.segment_path):
	for file in filenames:
		data_path = os.path.join(config.segmentGenerator.segment_path,file)
		# print(data_path)
		dfX = pandas.read_csv(data_path, sep=",",header=0)
		X_with_segments = dfX.as_matrix()

		X = np.concatenate((X_with_segments[:,1:src_name_col],\
			X_with_segments[:,src_name_col+1:dst_name_col],\
			X_with_segments[:,dst_name_col+1:fraud_col],X_with_segments[:,fraud_col+2:])\
			,axis=1)
		y = X_with_segments[:,fraud_col]

		if 1 not in y:
			continue

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
		while 1 not in y_train:
			# print("in while")
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

		y_train = np.expand_dims(y_train,axis=1)
		y_test = np.expand_dims(y_test,axis=1)
		# print('X_train shape')
		# print(X_train.shape)
		# print('y_train shape')
		# print(y_train.shape)

		train_data = np.concatenate((X_train,y_train),axis=1)
		test_data = np.concatenate((X_test,y_test),axis=1)

		if len(train_full)>0:
			train_full = np.concatenate((train_full,train_data),axis=0)
			test_full = np.concatenate((test_full,test_data),axis=0)
		else:
			train_full = train_data
			test_full = test_data

		file_train = os.path.join(config.segmentGenerator.train_segment_path,file)
		file_test = os.path.join(config.segmentGenerator.test_segment_path,file)
		writeArrayToCsv(train_data,file_train)
		writeArrayToCsv(test_data,file_test)

train_full_path = config.segmentGenerator.train_full_path
test_full_path = config.segmentGenerator.test_full_path
writeArrayToCsv(train_full,train_full_path)
writeArrayToCsv(test_full,test_full_path)