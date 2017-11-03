import time

import json

import pandas
import numpy as np
import csv

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


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

data_path = "./../../datasets/dataset_secondary.csv"

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

true_k, sc_vals = getBestCluster(X_trimmed_features,_min=2,_max=10)
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
data_path_primary = "./../../datasets/dataset_primary.csv"

print("Reading dataset...")
X_dataframe_pri = pandas.read_csv(data_path_primary, sep=",",header=0)
X_pri = X_dataframe_pri.as_matrix()
print("Read dataset")
col_names = list(X_dataframe_pri.columns.values)

X_with_segments = []
with open("./../../datasets/dataset_primary_segmented.csv","w") as f:
	# X_with_segments = np.concatenate((np.expand_dims(prediction,axis=1),X_pri),axis=1)
	for i in range(X_pri.shape[0]):
		X_with_segments.append(np.concatenate(([[seg_dict[X_pri[i,3]]]],np.expand_dims(X_pri[i,:],axis=0)),axis=1)[0])
	writer = csv.writer(f)
	writer.writerows(X_with_segments)

#this is to remove the newlines
with open("./../../datasets/dataset_primary_segmented.csv", "r") as f:
	lines = f.readlines()
	lines = [line for i,line in enumerate(lines) if i%2==0]

with open("./../../datasets/dataset_primary_segmented.csv","w") as f:
	header = "segment,step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,accountType,isFraud,isFlaggedFraud"
	f.write(header + "\n" + "".join(lines))

print("Saved segmented dataset")