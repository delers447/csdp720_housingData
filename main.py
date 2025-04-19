#! /usr/bin/python3

import celia
import dan
import sagar

import statistics

def ensemble_model(datum):
	results = []

	results.append(celia.model(datum))
	results.append(dan.model(datum))
	results.append(sagar.model(datum))

	return statistics.mean(results)

result = ensemble_model(100)
print(result)