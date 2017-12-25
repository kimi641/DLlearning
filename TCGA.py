from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import numpy as np
import tensorflow as tf
import collections

from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset',['data','target'])

TRAIN = "train.csv"
VALID = "valid.csv"
TEST = "test.csv"

def load_csv(filename,
	target_dtype,
	features_dtype,
	target_column=-1):
	"""load CSV with header"""
	with gfile.Open(filename) as csv_file:
		data_file = csv.reader(csv_file)
		header = next(data_file)
		n_features = len(header) - 1
		n_samples = sum(1 for row in data_file)
		print ( n_samples)
		print ( n_features)
		data = np.zeros((n_samples, n_features), dtype=features_dtype)
		target = np.zeros((n_samples,),dtype=target_dtype)
		for i, row in enumerate(data_file):
			target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
			data[i] = np.asarray(row, dtype=features_dtype)

	return Dataset(data=data,target=target)
		

def main():
	train_set = load_csv(
		filename=TRAIN,
		target_dtype=np.int,
		features_dtype=np.float32)
	valid_set = load_csv(
		filename=VALID,
		target_dtype=np.int,
		features_dtype=np.float32)

	feature_columns = [tf.feature_column.numeric_column("x", shape=[10551])]


	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
		hidden_units = [100,100,100],
		n_classes=14,
		model_dir="./TCGA_model")

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(train_set.data)},
		y=np.array(train_set.target),
		num_epochs=None,
		shuffle=True)
	
	classifier.train(input_fn=train_input_fn, steps=20000)
	
	valid_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(valid_set.data)},
		y=np.array(valid_set.target),
		num_epochs=1,
		shuffle=False)

	accuracy_score = classifier.evaluate(input_fn=valid_input_fn)["accuracy"]

	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

if __name__ == "__main__":
	main()

