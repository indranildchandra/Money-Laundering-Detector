import pandas
import json
import numpy as np
import csv
from random import randint

#####################################################################
#form primary data
######################################################################

data_path = "./../../datasets/filtered_data.csv"

print("Reading dataset")
X = pandas.read_csv(data_path, sep=",",header=0)
X = X.as_matrix()
print("Read dataset")


# csv_dataset_primary = np.ndarray((X.shape[0],8))
csv_dataset_primary = []

#col
step = 0
trans_type = 1
amount = 2
nameOrig = 3
oldbalanceOrg = 4
nameDest = 6
oldbalanceDest = 7
accountType = 8
isFraud = 9
isFlaggedFraud = 10

# print(csv_dataset_primary.shape, X.shape)
transfer = ["WIRE_IN", "WIRE_OUT"]
for i in range(X.shape[0]):
	arr = []
	arr.append(X[i,step])
	if X[i,trans_type] =="PAYMENT":
		arr.append("CREDIT")
	elif X[i,trans_type] =="TRANSFER":
		arr.append(transfer[randint(0,1)])
	else:
		arr.append(X[i,trans_type])
	arr.append(X[i,amount])
	arr.append(X[i,nameOrig])
	arr.append(X[i,oldbalanceOrg])
	arr.append(X[i,nameDest])
	arr.append(X[i,oldbalanceDest])
	if X[i,trans_type] == "TRANSFER":
		arr.append("FOREIGN")
	else:
		arr.append("DOMESTIC")

	arr.append(X[i,isFraud])
	arr.append(X[i,isFlaggedFraud])

	csv_dataset_primary.append(arr)

with open("./../../datasets/dataset_primary.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_dataset_primary)

#this is to remove the newlines
with open("./../../datasets/dataset_primary.csv", "r") as f:
	lines = f.readlines()
	lines = [line for i,line in enumerate(lines) if i%2==0]

with open("./../../datasets/dataset_primary.csv","w") as f:
	header = "step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,accountType,isFraud,isFlaggedFraud"
	f.write(header + "\n" + "".join(lines))
