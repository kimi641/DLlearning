#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf

FLAGS = None

TRAIN = "train.csv"
VALID = "valid.csv"
TEST = "test.csv"

from tensorflow.contib.learn.python.learn.datasets import base

def deepnn(x):

	"""deepnn for classifying cancer types by using log flod change
	Argvs:
	x: an input tensor with the dimensions (N_examples, 10551), where 10551
	represent gene numbers

	Return:
	A tuple (y, keep_prob). y is a tensor of shape (N_examples, 14), where 14
	represent 14 classes. keep_prob is a sclar placeholder for probability of
	drop out.
	"""
	with tf.name_scope('dnn1'):
		W_dnn1 = weight_variable([100551])
		b_dnn1 = bias_variable([10551])
		h_dnn1 = tf.nn.relu(dnn(x, W_dnn1) + b_dnn1)
	
	with tf.name_scope('dnn2'):
		W_dnn2 = weight_variable([100])
		b_dnn2 = bias_variable([100])
		h_dnn2 = tf.nn.relu(dnn(h_dnn1, W_dnn2) + b_dnn2)
	
	with tf.name_scope('dnn3'):
		W_dnn3 = weight_variable([100])
		b_dnn3 = bias_variable([100])
		h_dnn3 = tf.nn.relu(dnn(h_dnn2, W_dnn3) + b_dnn3)
	
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_drop = tf.nn.dropout(h_dnn3, keep_prob)
		
	with tf.name_scope('fc'):
		W_fc = weight_variable([100,14])
		b_fc = weight_varibale([14])

		y_head = tf.matmul(h_drop, W_fc) + b_fc
	return y_head, keep_prob



def dnn(x, W):
	return tf.matmul(x, W)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

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

class DataSet(object):
	
	def __init__(self,
			lfcs,
			labels,
			dtype=dtype.float32,
			)
	if dtype not in (dtypes.int,dytpes.float32):
		raise TypeError('Invalid input')
	assert lfcs.shape[0] == labels.shape[0],(
		'lfcs.shape: %s lables.shape: %s' % (lfcs.shape, labels.shape))
	
	self._num_expamples = lfcs.shape[0]
	self._lfcs = lfcs
	self._labels = labels
	self._epochs_completed = 0
	self._index_in_epoch = 0

	@property
	def lfcs(self):
		return self._lfcs
	
	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples
	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, shuffle=True)
		start = self._index_in_epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = numpy.arrange(self._num_examples)
			numpy.random.shuffle(perm0)
			self._images = self.images[perm0]
	
	
def input_data():
	train = Dataset()


def main(_):
	tcga = input_data()
	
	x = tf.placeholder(tf.float32, [None, 100551])
	
	y_ = tf.placeholder(tf.float32, [None, 14])

	y_head, keep_prob = deepnn(x)

	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entroy_with_logits(labels=y_,
									logits=y_head)
	cross_entropy = tf.reduce_mean(cross_entropy)

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	accuracy = tf.reduce_mean(correct_prediction)
	
	graph_location = tmpfile.mkdtemp()
	print ('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(20000):
			batch = tcga.train.next_batch(50)
			if i % 10 == 0:
				train_accuracy = accuracy.eval(feed_dict = {
					x: batch[0], y_: batch[1], keep_prob: 1.0})
				print ('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.tun(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		print ('test accuracy %g' % accuracy.eval(feed_dict={
			x: tcga.test.lfc, y_: tcga.test.labels, keep_: 1.0}))

	if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('--data_dir', type=str,
					default='/tmp/tensorflow/mnist/input_data',
					help='Directory for storing input data')
		FLAGS, unparsed = parser.parse_known_args()
		tf.app.run(main=main, argv=[sys.argv[0]]+ unparsed)
