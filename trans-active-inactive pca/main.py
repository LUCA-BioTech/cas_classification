import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

# Put train/test data (protein ID/sequence with ESMfold last layer embeddings) described in the paper in the same folder
# with the python file.

log_df = pd.read_csv('train.csv')
print(log_df)
e_list = list(log_df['embedding'])

feature_list = []

for e in e_list:
    a = e[1:-1].split(', ')
    aa = [float(i) for i in a]
    feature_list.append(aa)

print("the length of dim is", np.shape(e_list),np.shape(feature_list))

label = list(log_df['label'])

# Manually change the number of principal components from 2-15

pca = PCA(n_components=7)
pca = pca.fit(feature_list)
x_new = pca.transform(feature_list)

# Use x_new as PCA for covariates to predict trans-active/inactive label for proteins.
# Use feature_list as embedding for covariate to predict trans-active/inactive label for proteins.
# Comment the other line, use only one line for train_data.

train_data = pd.DataFrame(x_new)
# train_data = pd.DataFrame(feature_list)

train_data['class'] = label
print(train_data)

# Use test set to examine the accuracy. Test set has labels for trans-active/inactive.

log_df_test = pd.read_csv('test.csv')

e_list_test = list(log_df_test['embedding'])

feature_list_test = []

for e in e_list_test:
    a = e[1:-1].split(', ')
    aa = [float(i) for i in a]
    feature_list_test.append(aa)

label_test = list(log_df_test['label'])

x_new_test = pca.transform(feature_list_test)

# Use x_new_test as PCA for covariates to predict trans-active/inactive label for proteins.
# Use feature_list_test as embedding for covariate to predict trans-active/inactive label for proteins.
# Comment the other line, use only one line for train_data. Must be the same as training process.

test_data = pd.DataFrame(x_new_test)
# test_data = pd.DataFrame(feature_list_test)

test_data['class'] = label_test

predictor = TabularPredictor(label='class').fit(train_data, time_limit=120)  # Fit models for 120s
leaderboard = predictor.leaderboard(test_data)

# Record the outputs into csv files. Change file names for other outputs.
# Here we use 7 PCA as an output example.

leaderboard.to_csv('result_pca7.csv', index=False)

