#! /usr/bin/python3

import celia
import dan_real as dan
import sagar

import pandas as pd
import statistics

#https://www.kaggle.com/competitions/home-data-for-ml-course/code

test_file_path = "test.csv"
test_data = pd.read_csv(test_file_path)
ids = test_data.pop('Id')

def ensemble_model(data):
	results = []

	#results.append(celia.model(datum))
	#results.append(sagar.model(datum))
	pred1 = dan.model1(data)
	pred2 = dan.model2(data)
	pred3 = dan.model3(data)

	return (pred1, pred2, pred3)
	#return statistics.mean(results)



pred1, pred2, pred3 = ensemble_model(test_data)

output = pd.DataFrame({'Id': ids,
                       'SalePrice1': pred1.squeeze(),
                       'SalePrice2': pred2.squeeze(),
                       'SalePrice3': pred3.squeeze()})

print(output)