#! /usr/bin/python3

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import tf_keras
import tensorflow_decision_forests as tfdf
import pickle

#model = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
#model.load_weights('dan_model1')

#model = tf_keras.models.load_model('dan_model1')

'''
def model1(data):
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(data, task = tfdf.keras.Task.REGRESSION)
	
	model = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
	model.load_weights('dan_model1')

	predictions = model.predict(test_ds)
	return predictions
'''
'''
def model1(data):
	# Convert pandas DataFrame to tf.data.Dataset (features only)
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(data, task=tfdf.keras.Task.REGRESSION)

	# Load the entire trained model (not just weights)
	model = tf.keras.models.load_model('dan_model1.keras')

	# Predict on the dataset
	predictions = model.predict(test_ds)
    
	return predictions
'''

def model1(data):
    # Convert pandas DataFrame to tf.data.Dataset (features only)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(data, task=tfdf.keras.Task.REGRESSION)

    # Load the model saved in .keras format
    model = tf.keras.models.load_model('dan_model1.keras')

    # Prepare the dataset: The model expects a dictionary of features, not just a tensor
    #all_predictions = []
    #for example in test_ds:
    #    # Here, 'example' is a dictionary of features, you can pass it directly to the model
    #    predictions = model.predict(example)  # 'example' is the feature dictionary
    #    all_predictions.append(predictions)

    # Combine predictions from all batches
    #all_predictions = tf.concat(all_predictions, axis=0)

    return all_predictions

def model2(data):
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(data, task = tfdf.keras.Task.REGRESSION)

	model = tf.keras.models.load_model('dan_model2.keras')

	predictions = model.predict(test_ds)
	return predictions

def model3(data):
	test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(data, task = tfdf.keras.Task.REGRESSION)

	model = tf.keras.models.load_model('dan_model3.keras')

	predictions = model.predict(test_ds)
	return predictions