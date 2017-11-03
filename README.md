This Codebase is to prove the hypothesis that a solution powered by Machine Learning and Behaviour Analytics will find…
-> currently invisible transaction behaviour
-> aberrations in transactions
-> reduce review operations cost by lowering the number of False Positive alerts 
without using current framework of static rule based alert generation process

#Steps to run the Code
1. Download the Golden Dataset from -> https://www.kaggle.com/ntnu-testimon/paysim1/data
2. Filter the Golden Dataset and scale down to only few entities [unique nameOrig and nameDest] with more than 40 transaction records for each entity.
Code -> machine-learning-layer/src/data_engineering/filteredDataGenerator.py
Output -> machine-learning-layer/datasets/filtered_data.csv
3. Transform the attributes from the Golden Dataset to Atrributes that can be used for our modelling. The features are hypothesised and derived with an aim to capture the Volume trends and Velocity trends in the transactions.
Code -> machine-learning-layer/src/data_engineering/primaryDataGenerator.py
Output -> machine-learning-layer/datasets/dataset_primary.csv
4. Transform the attributes into features that can be feeded to the Machine Learning algorithm.
Code -> machine-learning-layer/src/data_engineering/secondaryDataGenerator.py
Output -> machine-learning-layer/datasets/dataset_secondary.csv
5. Identify the most useful features and rank them depending on their entropy, i.e, a features that contributes more to the decision whether a transaction is fraudulent or not, gets a higher rank as compared to other features. This step does a basic sanity check whether the features that are being used are mathematically valid or not; the entropy values should be more close to 1 than to 0 to prove the hypothesis right. Also it gives flexibility to assign weightages to the features when feeded into the final classifier algorithm.
Algorithm -> Decision Tree: C4.5/ID3
Code -> machine-learning-layer/src/behavioural_segmentation/featureRanker.py
6. Group the entities who have similar transaction patterns into a single segment. 
Algorithm -> K-means
Code -> machine-learning-layer/src/behavioural_segmentation/segmentGenerator.py
Output -> machine-learning-layer/datasets/dataset_primary_segmented.csv
7. Classify each transaction as Fraudulent or not depending on the trained model
Algorithm -> SVM [lower f1 score] and Decision Tree [higher f1 score]
Code -> src/fraudulent_transaction_classifier/svmClassifier.py ; src/fraudulent_transaction_classifier/decisionTreeClassifier.py
Trained Model -> models/tree_classifier_model.dat


The attributes in the datasets are described below: 
1. Golden Data Source - Paysim1 [PS_20174392719_1491204439457_log.csv]:
step -> Day when the transaction happened [Assume 01/01/2017 as Day 0, then step=5 indicates the transaction was carried out on 05/01/2017]
type -> Indicates the Type of Transaction
amount -> Indicates the Amount that was transferred
nameOrig -> Indicates the name of the entity who transferred the amount
oldbalanceOrg -> Indicates the Balance of the entity who transferred the amount, before the transaction happened
nameDest -> Indicates the name of the entity who received the amount
oldbalanceDest -> Indicates the Balance of the entity who received the amount, before the transaction happened
newbalanceDest -> Indicates the Balance of the entity who received the amount, after the transaction happened
isFraud -> Indicates 1 if its a genuine case of Fraudulent transaction
isFlaggedFraud -> Indicates 1 if its tagged as a Fraudulent transaction by the static rule based methods

2. filtered_data.csv
-> Same as that of Golden Data Source - Paysim1

3. dataset_primary.csv
step -> Day when the transaction happened [Assume 01/01/2017 as Day 0, then step=5 indicates the transaction was carried out on 05/01/2017]
trans_type -> Indicates the Type of Transaction
amount -> Indicates the Amount that was transferred
nameOrig -> Indicates the name of the entity who transferred the amount
oldbalanceOrg -> Indicates the Balance of the entity who transferred the amount, before the transaction happened
nameDest -> Indicates the name of the entity who received the amount
oldbalanceDest -> Indicates the Balance of the entity who received the amount, before the transaction happened
accountType -> Indicates the Type of Account, i.e. Domestic or Foreign
isFraud -> Indicates 1 if its a genuine case of Fraudulent transaction
isFlaggedFraud -> Indicates 1 if its tagged as a Fraudulent transaction by the static rule based methods

4. dataset_secondary.csv
entity -> Entity Name
incoming_domestic_amount_30 -> Transactions done to the entity's account in first 30 days, for domestic accounts
incoming_domestic_amount_60 -> Transactions done to the entity's account in first 60 days, for domestic accounts
incoming_domestic_amount_90 -> Transactions done to the entity's account in first 90 days, for domestic accounts
outgoing_domestic_amount_30 -> Transactions done from the entity's account in first 30 days, for domestic accounts
outgoing_domestic_amount_60 -> Transactions done from the entity's account in first 60 days, for domestic accounts
outgoing_domestic_amount_90 -> Transactions done from the entity's account in first 90 days, for domestic accounts
incoming_foreign_amount_30 -> Transactions done to the entity's account in first 30 days, for foreign accounts
incoming_foreign_amount_60 -> Transactions done to the entity's account in first 60 days, for foreign accounts
incoming_foreign_amount_90 -> Transactions done to the entity's account in first 90 days, for foreign accounts
outgoing_foreign_amount_30 -> Transactions done from the entity's account in first 30 days, for foreign accounts
outgoing_foreign_amount_60 -> Transactions done from the entity's account in first 60 days, for foreign accounts
outgoing_foreign_amount_90 -> Transactions done from the entity's account in first 90 days, for foreign accounts
incoming_domestic_count_30 -> Number of Transactions done to the entity's account in first 30 days, for domestic accounts
incoming_domestic_count_60 -> Number of Transactions done to the entity's account in first 60 days, for domestic accounts
incoming_domestic_count_90 -> Number of Transactions done to the entity's account in first 90 days, for domestic accounts
outgoing_domestic_count_30 -> Number of Transactions done from the entity's account in first 30 days, for domestic accounts
outgoing_domestic_count_60 -> Number of Transactions done from the entity's account in first 60 days, for domestic accounts
outgoing_domestic_count_90 -> Number of Transactions done from the entity's account in first 90 days, for domestic accounts
incoming_foreign_count_30 -> Number of Transactions done to the entity's account in first 30 days, for foreign accounts
incoming_foreign_count_60 -> Number of Transactions done to the entity's account in first 60 days, for foreign accounts
incoming_foreign_count_90 -> Number of Transactions done to the entity's account in first 90 days, for foreign accounts
outgoing_foreign_count_30 -> Number of Transactions done from the entity's account in first 30 days, for foreign accounts
outgoing_foreign_count_60 -> Number of Transactions done from the entity's account in first 60 days, for foreign accounts
outgoing_foreign_count_90 -> Number of Transactions done from the entity's account in first 90 days, for foreign account9
balance_difference_30 -> Difference in balance in the entity's account on 30th day and 1st day
balance_difference_60 -> Difference in balance in the entity's account on 60th day and 1st day
balance_difference_90 -> Difference in balance in the entity's account on 90th day and 1st day
isFraud -> =1 if the transacation is fraudulent, =0 if it's not a fraudulent transaction
