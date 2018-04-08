import pandas
import json
import numpy as np
import csv
from random import randint

import sys
sys.path.append('../..')
import config

###########################################################################
#write secondary file
###########################################################################
data_path = config.secondaryDataGenerator.data_path

print("Reading dataset")
X = pandas.read_csv(data_path, sep=",",header=0)
X = X.as_matrix()
print("Read dataset")

#col
step = 0
trans_type = 1
amount = 2
nameOrig = 3
oldbalanceOrg = 4
nameDest = 5
oldbalanceDest = 6
accountType = 7
isFraud = 8

#col
entity = 0
incommingDomestic30 = 1
incommingDomestic60 = 2
incommingDomestic90 = 3
outgoingDomestic30 = 4
outgoingDomestic60 = 5
outgoingDomestic90 = 6
incommingForeign30 = 7
incommingForeign60 = 8
incommingForeign90 = 9
outgoingForeign30 = 10
outgoingForeign60 = 11
outgoingForeign90 = 12
incoming_domestic_count_30 = 13
incoming_domestic_count_60 = 14
incoming_domestic_count_90 = 15
outgoing_domestic_count_30 = 16
outgoing_domestic_count_60 = 17
outgoing_domestic_count_90 = 18
incoming_foreign_count_30 = 19
incoming_foreign_count_60 = 20
incoming_foreign_count_90 = 21
outgoing_foreign_count_30 = 22
outgoing_foreign_count_60 = 23
outgoing_foreign_count_90 = 24
balance_difference_30 = 25
balance_difference_60 = 26
balance_difference_90 = 27
isFraudSec = 28

csv_dataset_secondary = []
entities_pos = {}
enititesDict = {}

def getSecRow(entity):
	return [entity,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

modified_primary_data = []
for i in range(X.shape[0]):
	source_entity = X[i,nameOrig]
	dest_entity = X[i,nameDest]

	source_pos = entities_pos.get(source_entity,-1)
	if source_pos == -1:
		pos = len(csv_dataset_secondary)
		entities_pos[source_entity] = pos
		source_pos = pos

		row = getSecRow(source_entity)

		csv_dataset_secondary.append(row)

	dest_pos = entities_pos.get(dest_entity,-1)
	if dest_pos == -1:
		pos = len(csv_dataset_secondary)
		entities_pos[dest_entity] = pos
		dest_pos = pos

		row = getSecRow(dest_entity)
		
		csv_dataset_secondary.append(row)

	transferAmountSource = 0
	transferAmountDest = 0

	if X[i,trans_type] == "CASH_IN" or X[i,trans_type] == "CREDIT":
		if X[i,step] <=30:
			csv_dataset_secondary[source_pos][incommingDomestic30] += X[i,amount]
			csv_dataset_secondary[dest_pos][outgoingDomestic30] += X[i,amount]
		if X[i,step] <=60:
			csv_dataset_secondary[source_pos][incommingDomestic60] += X[i,amount]
			csv_dataset_secondary[dest_pos][outgoingDomestic60] += X[i,amount]
		if X[i,step] <=90:
			csv_dataset_secondary[source_pos][incommingDomestic90] += X[i,amount] 
			csv_dataset_secondary[dest_pos][outgoingDomestic90] += X[i,amount]

		transferAmountSource = X[i,amount]
		transferAmountDest = -1*X[i,amount]

	if X[i,trans_type] == "CASH_OUT" or X[i,trans_type] == "DEBIT":
		if X[i,step] <=30:
			csv_dataset_secondary[source_pos][outgoingDomestic30] += X[i,amount]
			csv_dataset_secondary[dest_pos][incommingDomestic30] += X[i,amount]
		if X[i,step] <=60:
			csv_dataset_secondary[source_pos][outgoingDomestic60] += X[i,amount]
			csv_dataset_secondary[dest_pos][incommingDomestic60] += X[i,amount]
		if X[i,step] <=90:
			csv_dataset_secondary[source_pos][outgoingDomestic90] += X[i,amount] 
			csv_dataset_secondary[dest_pos][incommingDomestic90] += X[i,amount]

		transferAmountSource = -1*X[i,amount]
		transferAmountDest = X[i,amount]

	if X[i,trans_type] == "WIRE_IN":
		if X[i,step] <=30:
			csv_dataset_secondary[source_pos][incommingForeign30] += X[i,amount]
			csv_dataset_secondary[dest_pos][outgoingForeign30] += X[i,amount]
		if X[i,step] <=60:
			csv_dataset_secondary[source_pos][incommingForeign60] += X[i,amount]
			csv_dataset_secondary[dest_pos][outgoingForeign60] += X[i,amount]
		if X[i,step] <=90:
			csv_dataset_secondary[source_pos][incommingForeign90] += X[i,amount] 
			# print(dest_pos,outgoingForeign90,i,amount)
			csv_dataset_secondary[dest_pos][outgoingForeign90] += X[i,amount]

		transferAmountSource = X[i,amount]
		transferAmountDest = -1*X[i,amount]

	if X[i,trans_type] == "WIRE_OUT":
		if X[i,step] <=30:
			csv_dataset_secondary[source_pos][outgoingForeign30] += X[i,amount]
			csv_dataset_secondary[dest_pos][incommingForeign30] += X[i,amount]
		if X[i,step] <=60:
			csv_dataset_secondary[source_pos][outgoingForeign60] += X[i,amount]
			csv_dataset_secondary[dest_pos][incommingForeign60] += X[i,amount]
		if X[i,step] <=90:
			csv_dataset_secondary[source_pos][outgoingForeign90] += X[i,amount] 
			csv_dataset_secondary[dest_pos][incommingForeign90] += X[i,amount]

		transferAmountSource = -1*X[i,amount]
		transferAmountDest = X[i,amount]

	if enititesDict.get(source_entity,-1) == -1:
		enititesDict[source_entity] = {
										'day1Bal': X[i,oldbalanceOrg],
										'day30Bal': 0,
										'day60Bal': 0,
										'day90Bal': 0,
										'countIncomingDomestic30': 0,
										'countOutgoingDomestic30': 0,
										'countIncomingDomestic60': 0,
										'countOutgoingDomestic60': 0,
										'countIncomingDomestic90': 0,
										'countOutgoingDomestic90': 0,
										'countIncomingForeign30': 0,
										'countOutgoingForeign30': 0,
										'countIncomingForeign60': 0,
										'countOutgoingForeign60': 0,
										'countIncomingForeign90': 0,
										'countOutgoingForeign90': 0
										}

	if enititesDict.get(dest_entity,-1) == -1:
		enititesDict[dest_entity] = {
										'day1Bal': X[i,oldbalanceDest],
										'day30Bal': 0,
										'day60Bal': 0,
										'day90Bal': 0,
										'countIncomingDomestic30': 0,
										'countOutgoingDomestic30': 0,
										'countIncomingDomestic60': 0,
										'countOutgoingDomestic60': 0,
										'countIncomingDomestic90': 0,
										'countOutgoingDomestic90': 0,
										'countIncomingForeign30': 0,
										'countOutgoingForeign30': 0,
										'countIncomingForeign60': 0,
										'countOutgoingForeign60': 0,
										'countIncomingForeign90': 0,
										'countOutgoingForeign90': 0
									}

	incomingForSource = ["CASH_IN","CREDIT","WIRE_IN"]
	incomingForDest = ["CASH_OUT","DEBIT","WIRE_OUT"]
	outgoingForDest = incomingForSource
	outgoingForSource = incomingForDest

	#Assumming the steps are in order
	if X[i,step]<=30:
		enititesDict[source_entity]['day30Bal'] = transferAmountSource+X[i,oldbalanceOrg]
		enititesDict[dest_entity]['day30Bal'] = transferAmountDest+X[i,oldbalanceDest]
		if X[i,accountType] == "FOREIGN":
			if X[i,trans_type] in incomingForSource:
				enititesDict[source_entity]['countIncomingForeign30'] += 1
			else:
				enititesDict[source_entity]['countOutgoingForeign30'] += 1
		else:
			if X[i,trans_type] in incomingForDest:
				enititesDict[source_entity]['countIncomingDomestic30'] += 1
			else:
				enititesDict[source_entity]['countOutgoingDomestic30'] += 1


		if X[i,accountType] == "FOREIGN":
			if X[i,trans_type] in incomingForDest:
				enititesDict[dest_entity]['countIncomingForeign30'] += 1
			else:

				enititesDict[dest_entity]['countOutgoingForeign30'] += 1
		else:
			if X[i,trans_type] in incomingForDest:
				enititesDict[dest_entity]['countIncomingDomestic30'] += 1
			else:
				enititesDict[dest_entity]['countOutgoingDomestic30'] += 1

	if X[i,step]<=60:
		enititesDict[source_entity]['day60Bal'] = transferAmountSource+X[i,oldbalanceOrg]
		enititesDict[dest_entity]['day60Bal'] = transferAmountDest+X[i,oldbalanceDest]
		if X[i,accountType] == "FOREIGN":
			if X[i,trans_type] in incomingForSource:
				enititesDict[source_entity]['countIncomingForeign60'] += 1
			else:
				enititesDict[source_entity]['countOutgoingForeign60'] += 1
		else:
			if X[i,trans_type] in incomingForDest:
				enititesDict[source_entity]['countIncomingDomestic60'] += 1
			else:
				enititesDict[source_entity]['countOutgoingDomestic60'] += 1

		if X[i,accountType] == "FOREIGN":
			if X[i,trans_type] in incomingForDest:
				enititesDict[dest_entity]['countIncomingForeign60'] += 1
			else:
				enititesDict[dest_entity]['countOutgoingForeign60'] += 1
		else:
			if X[i,trans_type] in incomingForDest:
				enititesDict[dest_entity]['countIncomingDomestic60'] += 1
			else:
				enititesDict[dest_entity]['countOutgoingDomestic60'] += 1

	if X[i,step]<=90:
		enititesDict[source_entity]['day90Bal'] = transferAmountSource+X[i,oldbalanceOrg]
		enititesDict[dest_entity]['day90Bal'] = transferAmountDest+X[i,oldbalanceDest]
		if X[i,accountType] == "FOREIGN":
			if X[i,trans_type] in incomingForSource:
				enititesDict[source_entity]['countIncomingForeign90'] += 1
			else:
				enititesDict[source_entity]['countOutgoingForeign90'] += 1
		else:
			if X[i,trans_type] in incomingForDest:
				enititesDict[source_entity]['countIncomingDomestic90'] += 1
			else:
				enititesDict[source_entity]['countOutgoingDomestic90'] += 1

		if X[i,accountType] == "FOREIGN":
			if X[i,trans_type] in incomingForDest:
				enititesDict[dest_entity]['countIncomingForeign90'] += 1
			else:
				enititesDict[dest_entity]['countOutgoingForeign90'] += 1
		else:
			if X[i,trans_type] in incomingForDest:
				enititesDict[dest_entity]['countIncomingDomestic90'] += 1
			else:
				enititesDict[dest_entity]['countOutgoingDomestic90'] += 1


	csv_dataset_secondary[source_pos][balance_difference_30] = enititesDict[source_entity]['day30Bal'] - enititesDict[source_entity]['day1Bal']
	csv_dataset_secondary[source_pos][balance_difference_60] = enititesDict[source_entity]['day60Bal'] - enititesDict[source_entity]['day1Bal']
	csv_dataset_secondary[source_pos][balance_difference_90] = enititesDict[source_entity]['day90Bal'] - enititesDict[source_entity]['day1Bal'] 



	csv_dataset_secondary[source_pos][incoming_domestic_count_30] = enititesDict[source_entity]['countIncomingDomestic30']
	csv_dataset_secondary[source_pos][outgoing_domestic_count_30] = enititesDict[source_entity]['countOutgoingDomestic30']
	csv_dataset_secondary[source_pos][incoming_domestic_count_60] = enititesDict[source_entity]['countIncomingDomestic60']
	csv_dataset_secondary[source_pos][outgoing_domestic_count_60] = enititesDict[source_entity]['countOutgoingDomestic60']
	csv_dataset_secondary[source_pos][incoming_domestic_count_90] = enititesDict[source_entity]['countIncomingDomestic90']
	csv_dataset_secondary[source_pos][outgoing_domestic_count_90] = enititesDict[source_entity]['countOutgoingDomestic90'] 
	csv_dataset_secondary[source_pos][incoming_foreign_count_30] = enititesDict[source_entity]['countIncomingForeign30']
	csv_dataset_secondary[source_pos][outgoing_foreign_count_30] = enititesDict[source_entity]['countOutgoingForeign30']
	csv_dataset_secondary[source_pos][incoming_foreign_count_60] = enititesDict[source_entity]['countIncomingForeign60']
	csv_dataset_secondary[source_pos][outgoing_foreign_count_60] = enititesDict[source_entity]['countOutgoingForeign60']
	csv_dataset_secondary[source_pos][incoming_foreign_count_90] = enititesDict[source_entity]['countIncomingForeign90']
	csv_dataset_secondary[source_pos][outgoing_foreign_count_90] = enititesDict[source_entity]['countOutgoingForeign90'] 

	csv_dataset_secondary[source_pos][isFraudSec] = csv_dataset_secondary[source_pos][isFraudSec] or X[i,isFraud] 

	csv_dataset_secondary[dest_pos][incoming_domestic_count_30] = enititesDict[dest_entity]['countIncomingDomestic30']
	csv_dataset_secondary[dest_pos][outgoing_domestic_count_30] = enititesDict[dest_entity]['countOutgoingDomestic30']
	csv_dataset_secondary[dest_pos][incoming_domestic_count_60] = enititesDict[dest_entity]['countIncomingDomestic60']
	csv_dataset_secondary[dest_pos][outgoing_domestic_count_60] = enititesDict[dest_entity]['countOutgoingDomestic60']
	csv_dataset_secondary[dest_pos][incoming_domestic_count_90] = enititesDict[dest_entity]['countIncomingDomestic90']
	csv_dataset_secondary[dest_pos][outgoing_domestic_count_90] = enititesDict[dest_entity]['countOutgoingDomestic90'] 
	csv_dataset_secondary[dest_pos][incoming_foreign_count_30] = enititesDict[dest_entity]['countIncomingForeign30']
	csv_dataset_secondary[dest_pos][outgoing_foreign_count_30] = enititesDict[dest_entity]['countOutgoingForeign30']
	csv_dataset_secondary[dest_pos][incoming_foreign_count_60] = enititesDict[dest_entity]['countIncomingForeign60']
	csv_dataset_secondary[dest_pos][outgoing_foreign_count_60] = enititesDict[dest_entity]['countOutgoingForeign60']
	csv_dataset_secondary[dest_pos][incoming_foreign_count_90] = enititesDict[dest_entity]['countIncomingForeign90']
	csv_dataset_secondary[dest_pos][outgoing_foreign_count_90] = enititesDict[dest_entity]['countOutgoingForeign90'] 

	csv_dataset_secondary[dest_pos][isFraudSec] = csv_dataset_secondary[dest_pos][isFraudSec] or X[i,isFraud] 

	#add extra columns in the primary dataset
	mod_row = []
	for x in X[i]:
		mod_row.append(x)
	for el in csv_dataset_secondary[source_pos][1:-1]:
		mod_row.append(el)
	for el in csv_dataset_secondary[dest_pos][1:-1]:
		mod_row.append(el)
	modified_primary_data.append(mod_row)

#write secondary data
with open(config.secondaryDataGenerator.out_path, "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_dataset_secondary)

#this is to remove the newlines
with open(config.secondaryDataGenerator.out_path, "r") as f:
	lines = f.readlines()
	lines = [line for i,line in enumerate(lines)]

with open(config.secondaryDataGenerator.out_path,"w") as f:
	header = """entity,incoming_domestic_amount_30,incoming_domestic_amount_60,incoming_domestic_amount_90,\
outgoing_domestic_amount_30,outgoing_domestic_amount_60,outgoing_domestic_amount_90,\
incoming_foreign_amount_30,incoming_foreign_amount_60,incoming_foreign_amount_90,\
outgoing_foreign_amount_30,outgoing_foreign_amount_60,outgoing_foreign_amount_90,\
incoming_domestic_count_30,incoming_domestic_count_60,incoming_domestic_count_90,\
outgoing_domestic_count_30,outgoing_domestic_count_60,outgoing_domestic_count_90,\
incoming_foreign_count_30,incoming_foreign_count_60,incoming_foreign_count_90,\
outgoing_foreign_count_30,outgoing_foreign_count_60,outgoing_foreign_count_90,\
balance_difference_30,balance_difference_60,balance_difference_90,\
isFraud"""
	f.write(header + "\n" + "".join(lines))


#write modified primary data
with open(config.secondaryDataGenerator.out_path_primary, "w") as f:
    writer = csv.writer(f)
    writer.writerows(modified_primary_data)

#this is to remove the newlines
with open(config.secondaryDataGenerator.out_path_primary, "r") as f:
	lines = f.readlines()
	lines = [line for i,line in enumerate(lines)]

with open(config.secondaryDataGenerator.out_path_primary,"w") as f:
	header = """step,trans_type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,accountType,\
isFraud,isFlaggedFraud,incoming_domestic_amount_30_src,incoming_domestic_amount_60_src,incoming_domestic_amount_90_src,\
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