#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Training the Model
#========================================================================================================

from optparse import OptionParser
import sys
import time
import gzip
import tensorflow as tf
from Region import *
import Window2Tensor
import Models
import os

BATCH_SIZE = 64

Window_Size = (WIDTH * (HEIGHT+1) * 3)

log_dir = './CKPT'

def Loss(logits, labels):
	labels = tf.to_int32(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits, name = 'xentropy')
	return tf.reduce_mean(cross_entropy)

def training(loss, learning_rate):
	tf.summary.scalar('loss', loss)
	learning_rate = 1 * learning_rate
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess, eval_correct, tensor_placeholder, labels_placeholder, data_set):
	true_count = 0
	steps_per_epoch = BATCH_SIZE // BATCH_SIZE
	num_examples = steps_per_epoch * BATCH_SIZE
	for step in xrange(steps_per_epoch):
		
		tensor_feed, labels_feed = data_set.read_batch()
		feed_dict = {tensor_placeholder: tensor_feed, labels_placeholder: labels_feed}
		true_count += sess.run(eval_correct, feed_dict = feed_dict)
	precision = float(true_count) / num_examples
	print '\tNum examples: %d\tNum correct: %d\tPrecision @ 1: %.04f' % (num_examples, true_count, precision)

def do_eval_on_Testing(sess, eval_correct, testing_tensor_pl, testing_label_pl, Testing_tensor, Testing_labels):
	true_count = 0
	steps_per_epoch = 2000 // BATCH_SIZE
	num_examples = steps_per_epoch * BATCH_SIZE
	for step in xrange(steps_per_epoch):
		tensor = Testing_tensor[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
		label = Testing_labels[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
		feed_dict = {testing_tensor_pl: tensor, testing_label_pl: label}
		true_count += sess.run(eval_correct, feed_dict = feed_dict)
	precision = float(true_count) / 2000 
	print '\tNum examples: %d\tNum correct: %d\tPrecision @ 1: %.04f' % (2000, true_count, precision)
	


def runTraining(TrainingData,TestingData):
	max_steps = 10000
	print 'Open Training Data set at %s ....' % TrainingData
	TrainingData = gzip.open(TrainingData,'rb')
	print 'Open Testing Data set at %s ....' % TestingData
	TestingData = gzip.open(TestingData,'rb')
	data_sets_training = Window2Tensor.Data_Reader(TrainingData,batch_size=BATCH_SIZE)
	data_sets_testing = Window2Tensor.Data_Reader(TestingData,batch_size=2000) 
	
	print '\nLaunch Tensorflow and Training\n'
	with tf.Graph().as_default():
		tensor_placeholder, labels_placeholder = Window2Tensor.placeholder_inputs(BATCH_SIZE)
		
		testing_tensor_pl, testing_label_pl = Window2Tensor.placeholder_inputs(BATCH_SIZE) 
		
		Testing_tensor, Testing_label = data_sets_testing.read_batch()
		
		
		logits = Models.FullyConnectNN(tensor_placeholder, 10000, 5000, 3000, 1000, 500, 200, 50, 10)
		loss = Loss(logits, labels_placeholder)
		train_op = training(loss, learning_rate=0.005)
		eval_correct = evaluation(logits, labels_placeholder)
		summary = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		sess.run(init)

		for step in xrange(max_steps):
			start_time = time.time()
			feed_dict = Window2Tensor.fill_feed_dict(data_sets_training, tensor_placeholder, labels_placeholder)

			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			duration = time.time() - start_time
			if step % 10 == 0:
				print 'Step %d loss = %.3f (%.3f sec)' % (step, loss_value, duration)
				summary_str = sess.run(summary, feed_dict = feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			if (step + 1) % 100 == 0 or (step + 1) == max_steps:
				checkpoint_file = os.path.join(log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step = step)
				print 'Training Data Eval:'
				do_eval(sess, eval_correct, tensor_placeholder, labels_placeholder, data_sets_training)
				#print 'Validation Data Eval:'
				#do_eval(sess, eval_correct, tensor_placeholder, labels_placeholder, data_sets.validation)
				print 'Testing Data Eval:'
			#	do_eval_on_Testing(sess, eval_correct, testing_tensor_pl, testing_label_pl, Testing_tensor, Testing_label)
				do_eval_on_Testing(sess, eval_correct, tensor_placeholder, labels_placeholder, Testing_tensor, Testing_label)


def GetOptions():
	parser = OptionParser()
	parser.add_option('-','--',dest = '', metavar = '', help = '')
	(options,args) = parser.parse_args()
	
	return

def main(_):
	TrainingData = '/home/local/users/jw/TensorFlowCaller/Nebraska_NA12878_HG001_TruSeq_Exome/sample_1/windows_training.txt.gz'
	TestingData = '/home/local/users/jw/TensorFlowCaller/Nebraska_NA12878_HG001_TruSeq_Exome/sample_1/windows_testing.txt.gz'
	runTraining(TrainingData, TestingData)
	return

if __name__=='__main__':
	tf.app.run(main=main, argv = [sys.argv[0]])
