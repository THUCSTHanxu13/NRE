#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from init import *
bags_hit = 0.0
bags_sum = 0.0
loss_sum = 0.0
class Config(object):
	def __init__(self):
		self.sequence_size = LenLimit
		self.num_classes = len(RelationList)
		self.num_words = len(VocabList)
		self.num_positions = PositionLimit + PositionLimit + 1
		self.word_size = 100
		self.position_size = 5
		self.embedding_size = self.word_size + self.position_size
		self.filter_size = 3
		self.num_filters = 230
		self.relation_size = 230
		self.dropout_keep_prob = 0.5

class CNN(object):

	def __init__(self, config):
		sequence_size = config.sequence_size
		num_classes = config.num_classes
		num_words = config.num_words
		num_positions = config.num_positions

		embedding_size = config.embedding_size
		word_size = config.word_size
		position_size = config.position_size
		relation_size = config.relation_size

		filter_size = config.filter_size
		num_filters = config.num_filters
		dropout_keep_prob = config.dropout_keep_prob

		self.input_x = tf.placeholder(tf.int32, [None, sequence_size], name = "input_x")
		self.input_p = tf.placeholder(tf.int32, [None, sequence_size], name = "input_p")
		self.input_r = tf.placeholder(tf.int32, [1, 1], name = "input_r")
		self.input_y = tf.placeholder(tf.float32, [1, num_classes], name = "input_y")
		l2_loss = tf.constant(0.0)
		with tf.name_scope("embedding-lookup"):
			self.word_embeddings = tf.Variable(word_embeddings, name="word_embeddings")
			#print self.word_embeddings.get_shape()
			self.position_embeddings = tf.get_variable("position_embeddings", [num_positions, position_size])
			#print self.position_embeddings.get_shape()
			self.relation_embeddings = tf.get_variable("relation_embeddings", [num_classes, relation_size])
			#print self.relation_embeddings.get_shape()
			self.attention = tf.get_variable("attention", [num_filters, relation_size])
			#print self.attention.get_shape()
			self.x_initial = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)
			self.x_p = tf.nn.embedding_lookup(self.position_embeddings, self.input_p)
			self.x = tf.expand_dims(tf.concat(2, [self.x_initial, self.x_p]), -1) #(?, 140, 105, 1)
			self.r = tf.nn.embedding_lookup(self.relation_embeddings, self.input_r) #(1, 230)
			l2_loss += tf.nn.l2_loss(self.attention)

		with tf.name_scope("conv-maxpool"):
			W = tf.Variable(tf.truncated_normal(shape = [filter_size, embedding_size, 1, num_filters], stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
			conv = tf.nn.conv2d(self.x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
			h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
			self.y = tf.nn.max_pool(h, ksize=[1, sequence_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
			#print self.y.get_shape()
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.y = tf.reshape(self.y, [-1, num_filters])
		#print self.y.get_shape()
		with tf.name_scope('attention'):
			self.r = tf.reshape(self.r, [230, -1])
			#print self.r.get_shape()
			e = tf.matmul(tf.matmul(self.y, self.attention), self.r)
			alpha = tf.nn.softmax(e)
			#Attention = Attentiton * tf.reshape(self.r, [-1, relation_size])
			#Attentiton = tf.reduce_sum(Attentiton, 1, keep_dims = True)
			#Attentiton = tf.reshape(Attentiton, [-1, instance_size])
			#Attentiton = tf.nn.softmax(Attentiton)
			#Attentiton = tf.reshape(Attentiton, [-1, instance_size, 1])
			alpha_reshape = tf.reshape(alpha, [1, -1])
			self.y_attention = tf.matmul(alpha_reshape, self.y)
			print self.y_attention.get_shape()

		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.y_attention, dropout_keep_prob)
 		with tf.name_scope("output"):
			softmax_w = tf.get_variable("softmax_w", [num_filters, num_classes])
			softmax_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="softmax_b")
			self.scores = tf.nn.xw_plus_b(self.h_drop, softmax_w, softmax_b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")
			l2_loss += tf.nn.l2_loss(softmax_w)
			l2_loss += tf.nn.l2_loss(softmax_b)
			print self.scores.get_shape()
			self.weights = tf.nn.softmax(self.scores)
			self.weight = tf.matmul(self.weights, tf.transpose(self.input_y))
			print self.weight.get_shape()
		with tf.name_scope("loss"):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss = tf.reduce_mean(cross_entropy) + l2_lambda * l2_loss

		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

if __name__ == "__main__":
	path = '/Users/tcc/Documents/sina/'
	np.random.seed(0)
	tf.set_random_seed(0)
	x_train, p_train, y_train, r_train, bags_train, x_test, p_test, y_test, r_test, bags_test = readFromFile(path)
	config = Config()
	with tf.Graph().as_default():
		conf = tf.ConfigProto()
		#conf.gpu_options.per_process_gpu_memory_fraction = 0.5
		sess = tf.Session(config=conf)
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				m = CNN(config = config)
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(0.001)
			grads_and_vars = optimizer.compute_gradients(m.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			sess.run(tf.initialize_all_variables())

			def train_step(x_batch, p_batch, y_batch, r_batch):
				global bags_sum, bags_hit, loss_sum
				feed_dict = {
					m.input_x: x_batch,
					m.input_p: p_batch,
					m.input_r: r_batch,
					m.input_y: y_batch
				}
				_, step, loss, accuracy = sess.run(
					[train_op, global_step, m.loss, m.accuracy], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				bags_sum += 1.0
				loss_sum += loss
				if accuracy > 0.5:
					bags_hit += 1.0
				if bags_sum % 100 == 0:
					print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_sum/bags_sum, bags_hit/bags_sum))
					bags_sum = 0
					bags_hit = 0
					loss_sum = 0
			def test_step(x_batch, p_batch, y_batch, r_batch):
				"""
				Evaluates model on a test set
				"""
				nc = len(RelationList)
				rs = range(nc)
				ys = np.eye(nc)
				ws = np.zeros([nc])
				for i in range(nc):
					feed_dict = {
						m.input_x: x_batch,
						m.input_p: p_batch,
						m.input_r: np.array([[rs[i]]]),
						m.input_y: np.array([ys[i]])
					}
					step, loss, accuracy, tmp_w = sess.run(
						[global_step, m.loss, m.accuracy, m.weight],
						feed_dict)
					ws[i] = tmp_w[0]
				#print ws
				prediction = np.argmax(ws)
				#print prediction
				if prediction == r_batch[0][0]:
					return 1.0
				return 0.0
			loop = 0
			while(True):
				print 'Looping ', loop
				batches = batch_iter(x_train, p_train, y_train, r_train, bags_train)
				for batch in batches:
					x_batch, p_batch, y_batch, r_batch = batch
					train_step(np.array(x_batch), np.array(p_batch), np.array(y_batch), np.array(r_batch))
				batches_test = batch_iter(x_test, p_test, y_test, r_test, bags_test)
				hit_test = 0.0
				sum_test = 0.0
				for batch in batches_test:
					sum_test += 1.0
					x_batch, p_batch, y_batch, r_batch = batch
					hit_test += test_step(np.array(x_batch), np.array(p_batch), np.array(y_batch), np.array(r_batch))
				print 'Loop '+str(loop)+' Accuracy: ', hit_test/sum_test
				loop += 1
