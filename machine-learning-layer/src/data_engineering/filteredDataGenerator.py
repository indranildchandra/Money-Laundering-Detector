import pandas
import json
import numpy as np
import csv
from random import randint

data_path = "./../../datasets/golden data source - paysim1/PS_20174392719_1491204439457_log.csv"

print("Reading dataset...")
X = pandas.read_csv(data_path, sep=",",header=0)
X = X.as_matrix()
print("Read dataset")

##############################################################################
#check unique entries
##############################################################################
nameOrigCol = 3
nameDestCol = 6

nameOrig = []
nameDest = []
nameCount = {}

namesWithMoreThanOneOccurrence = []


print()
for name in X[:,nameOrigCol]:
	if nameCount.get(name,-1) == -1:
		nameOrig.append(name)

		nameCount[name] = 1

	else:
		nameCount[name] += 1
		namesWithMoreThanOneOccurrence.append(name)

for name in X[:,nameDestCol]:
	if nameCount.get(name,-1) == -1:
		nameDest.append(name)

		nameCount[name] = 1

	else:
		nameCount[name] += 1
		namesWithMoreThanOneOccurrence.append(name)

countArr = []
count = 0
for attr, value in nameCount.items():
	if value>40:
		countArr.append(value)
		count += 1

median = np.median(countArr)
# print("median above 40")
# print(median)
# print("count")
# print(count)

###################################################################
#form the dataset for above median
###################################################################

csv_golden_data = []

for i in range(X.shape[0]):
	if nameCount.get(X[i,3],-1) > 40 or nameCount.get(X[i,6],-1) > 40:
		csv_golden_data.append(X[i,:])

with open("./../../datasets/filtered_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_golden_data)

with open("./../../datasets/filtered_data.csv", "r") as f:
	lines = f.readlines()
	lines = [line for i,line in enumerate(lines) if i%2==0]

	with open("./../../datasets/filtered_data.csv","w") as f1:
		f1.write("".join(lines))