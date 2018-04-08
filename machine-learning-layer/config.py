train = False
train_test_entire = False

class filteredDataGenerator:
	data_path = "./../../datasets/golden-data-source-paysim1/PS_20174392719_1491204439457_log.csv"
	if train:
		out_path = "./../../datasets/train_test_on_same_entities/filtered_data.csv"
	else:
		out_path = "./../../datasets/train_test_on_same_entities/filtered_data.csv"

class primaryDataGenerator:
	if train:
		data_path = "./../../datasets/train_test_on_same_entities/filtered_data.csv"
		out_path = "./../../datasets/train_test_on_same_entities/dataset_primary.csv"
	else:
		data_path = "./../../datasets/train_test_on_same_entities/filtered_data.csv"
		out_path = "./../../datasets/train_test_on_same_entities/dataset_primary.csv"

class secondaryDataGenerator:
	if train:
		data_path = "./../../datasets/train_test_on_same_entities/dataset_primary.csv"
		out_path = "./../../datasets/train_test_on_same_entities/dataset_secondary.csv"
		out_path_primary = "./../../datasets/train_test_on_same_entities/dataset_primary_velo_volu.csv"
	else:
		data_path = "./../../datasets/train_test_on_same_entities/dataset_primary.csv"
		out_path = "./../../datasets/train_test_on_same_entities/dataset_secondary.csv"
		out_path_primary = "./../../datasets/train_test_on_same_entities/dataset_primary_velo_volu.csv"

class featureRanker:
	if train:
		data_path = './../../datasets/train_test_on_same_entities/dataset_secondary.csv'
	else:
		data_path = './../../datasets/train_test_on_same_entities/dataset_secondary.csv'
	model_path = "./../../models/tree_classifier_model.dat"

class segmentGenerator:
	if train:
		out_path = "./../../datasets/train_test_on_same_entities/dataset_primary_segmented.csv"
		segment_path = "./../../datasets/train_test_on_same_entities/segments"
	else:
		out_path = "./../../datasets/train_test_on_same_entities/dataset_primary_segmented.csv"
		segment_path = "./../../datasets/train_test_on_same_entities/segments"
	

	train_full_path = '../../datasets/train_test_on_same_entities/entire_train/entire.csv'
	test_full_path = '../../datasets/train_test_on_same_entities/entire_test/entire.csv'
	
	train_segment_path = '../../datasets/train_test_on_same_entities/entire_train'
	test_segment_path = '../../datasets/train_test_on_same_entities/entire_test'

class decisionTree:
	if not train_test_entire:
		data_train_test_path = "./../../datasets/train_test_on_same_entities/segments"
		train_path = '../../datasets/train_test_on_same_entities/segments/train'
		test_path = '../../datasets/train_test_on_same_entities/segments/test'
	else:
		data_train_test_path = "./../../datasets/train_test_on_same_entities/entire_train"
		train_path = '../../datasets/train_test_on_same_entities/entire_train'
		test_path = '../../datasets/train_test_on_same_entities/entire_test'
	model_path = "./../../models/fraud_classifier_tree/"
	test_path_prefix = "./../../testCases/testTreeClassifier_"

class svmClassifier:
	model_path = "./../../models/fraud_classifier_svm/"
	test_path_prefix = "./../../testCases/testSvmClassifier_"