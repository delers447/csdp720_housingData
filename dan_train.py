#! /usr/bin/python3

import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf#Import-the-library

train_file_path = "train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))
dataset_df = dataset_df.drop('Id', axis=1)
print(dataset_df.head(3))

def show_metrics():
	print(dataset_df.info())
	print(dataset_df['SalePrice'].describe())
	plt.figure(figsize=(9, 8))
	sns.distplot(dataset_df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
	plt.show()

	list(set(dataset_df.dtypes.tolist()))
	df_num = dataset_df.select_dtypes(include = ['float64', 'int64'])
	print(df_num.head())
	df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
	plt.show()


def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print(f"{len(train_ds_pd)} examples in training, {len(valid_ds_pd)} examples in testing.")

label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)

print(tfdf.keras.get_all_models())

rf1 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
rf1.compile(metrics=["mse"]) 

rf2 = tfdf.keras.GradientBoostedTreesModel(task = tfdf.keras.Task.REGRESSION)
rf2.compile(metrics=["mse"]) 

rf3 = tfdf.keras.CartModel(task = tfdf.keras.Task.REGRESSION)
rf3.compile(metrics=["mse"]) 

rf1.fit(x=train_ds)
rf2.fit(x=train_ds)
rf3.fit(x=train_ds)

test_file_path = "test.csv"
test_data = pd.read_csv(test_file_path)
ids = test_data.pop('Id')

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_data,
    task = tfdf.keras.Task.REGRESSION)

preds1 = rf1.predict(test_ds)
preds2 = rf2.predict(test_ds)
preds3 = rf3.predict(test_ds)

output = pd.DataFrame({'Id': ids,
                       'SalePrice1': preds1.squeeze(),
                       'SalePrice2': preds2.squeeze(),
                       'SalePrice3': preds3.squeeze()})

print(output)


rf1.save('dan_model1.keras')
rf2.save('dan_model2.keras')
rf3.save('dan_model3.keras')

rf1.save_weights('dan_model1')
rf2.save_weights('dan_model2')
rf3.save_weights('dan_model3')

