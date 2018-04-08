import matplotlib.pyplot as plt
import pandas
import os

import sys
sys.path.append('../..')
import config

segments_path = config.segmentGenerator.segment_path

transaction_out = ['WIRE_OUT','CASH_OUT','DEBIT']
transaction_in = ['WIRE_IN','CASH_IN','CREDIT'] 
srcBalCol = 4
dstBalCol = 6
amtCol = 2
typeCol = 1

if config.train:
	out_path = './train'
else:
	out_path = './test'

for file in os.listdir(segments_path):
	data_path = os.path.join(segments_path,file)
	data_frame = pandas.read_csv(data_path, sep=",",header=0)
	data_mat = data_frame.as_matrix()
	data_mat_1 = data_mat.copy()
	data_mat_2 = data_mat.copy()
	data_mat_3 = data_mat.copy()
	data_mat_4 = data_mat.copy()

	x_0 = []
	x_1 = []
	y_0_src = []
	y_1_src = []
	y_0_dst = []
	y_1_dst = []
	y_0_in = []
	y_1_in = []
	y_0_out = []
	y_1_out = []
	c_0 = []
	c_1 = []
	for i in range(data_mat.shape[0]):
		if data_mat[i,-2]==0:
			x_0.append(data_mat[i,amtCol])

			if data_mat[i,typeCol] in transaction_out:
				y_0_out.append(data_mat[i,srcBalCol])
			else:
				y_0_out.append(data_mat[i,srcBalCol])

			if data_mat[i,typeCol] in transaction_out:
				y_0_in.append(data_mat[i,dstBalCol])
			else:
				y_0_in.append(data_mat[i,dstBalCol])

			y_0_src.append(data_mat[i,srcBalCol])

			y_0_dst.append(data_mat[i,dstBalCol])

			c_0.append('blue')
		else:
			x_1.append(data_mat[i,amtCol])
			
			if data_mat[i,typeCol] in transaction_out:
				y_1_out.append(data_mat[i,srcBalCol])
			else:
				y_1_out.append(data_mat[i,srcBalCol])

			if data_mat[i,typeCol] in transaction_out:
				y_1_in.append(data_mat[i,dstBalCol])
			else:
				y_1_in.append(data_mat[i,dstBalCol])

			y_1_src.append(data_mat[i,srcBalCol])

			y_1_dst.append(data_mat[i,dstBalCol])

			c_1.append('red')

	x = x_0 + x_1
	c = c_0 + c_1

	#plot wrt src balance
	y = y_0_src + y_1_src

	plt.scatter(x,y,c=c)
	plt.xlabel('Transaction Amount')
	plt.ylabel('Account balance of origin at the time of transaction')

	file_name = os.path.join(out_path,'src_bal',file.split('.')[0]+'.jpg')
	plt.savefig(file_name)

	#plot outgoing
	y = y_0_out + y_1_out

	plt.scatter(x,y,c=c)
	plt.xlabel('Amount outgoing from account')
	plt.ylabel('Account balance at the time of transaction')

	file_name = os.path.join(out_path,'outgoing',file.split('.')[0]+'.jpg')
	plt.savefig(file_name)

	#plot incoming
	y = y_0_in + y_1_in

	plt.scatter(x,y,c=c)
	plt.xlabel('Amount incoming to account')
	plt.ylabel('Account balance at the time of transaction')

	file_name = os.path.join(out_path,'incoming',file.split('.')[0]+'.jpg')
	plt.savefig(file_name)

	#plot wrt dst balance
	y = y_0_dst + y_1_dst

	plt.scatter(x,y,c=c)
	plt.xlabel('Transaction Amount')
	plt.ylabel('Account balance of destination at the time of transaction')

	file_name = os.path.join(out_path,'dst_bal',file.split('.')[0]+'.jpg')
	plt.savefig(file_name)