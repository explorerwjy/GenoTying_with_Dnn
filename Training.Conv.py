#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Training The ConvNet for Tensor Caller
#========================================================================================================

import argparse
from datetime import datetime
import time
import os 
import threading
import numpy as np
import tensorflow as tf
import Models
from Input import *
import sys
sys.stdout = sys.stderr

BATCH_SIZE=FLAGS.batch_size
log_dir = FLAGS.log_dir
max_steps = FLAGS.max_steps

def train_2():
	"""Train TensorCaller for a number of steps."""
	with tf.Graph().as_default():
		print "Locating Data File"
		TrainingData = gzip.open(FLAGS.TrainingData,'rb')
		TestingData = gzip.open(FLAGS.TestingData,'rb')
		data_sets_training = Data_Reader(TrainingData, batch_size=BATCH_SIZE)
		data_sets_testing = Data_Reader(TestingData, batch_size=BATCH_SIZE) 
		print "Training Data @%s; \nTesting Data @%s" % (os.path.abspath(FLAGS.TrainingData), os.path.abspath(FLAGS.TestingData))
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Get Tensors and labels for Training data.
		#tensors, labels = Models.inputs(FLAGS.data_file)

				
		# Build a Graph that computes the logits predictions from the
		# inference model.
		tensor_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
		convnets = Models.ConvNets()
		logits = convnets.Inference(tensor_placeholder)

		# Calculate loss.
		loss = convnets.loss(logits, labels_placeholder)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnets.Train(loss, global_step)

		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""

			def begin(self):
				self._step = -1

			def before_run(self, run_context):
				self._step += 1
				self._start_time = time.time()
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				duration = time.time() - self._start_time
				loss_value = run_values.results
				if self._step % 100 == 0: # Output Loss Every 100 Steps Training
					num_examples_per_step = FLAGS.batch_size
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = float(duration)

					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
					print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
				#if self._step % 1000 == 0: # Output Loss of Evauation Data Every 100 Steps



	with tf.train.MonitoredTrainingSession(
		checkpoint_dir=FLAGS.train_dir, 
		hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()],
		config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
		while not mon_sess.should_stop():
			feed_dict = fill_feed_dict(data_sets_training, tensor_placeholder, labels_placeholder)
			#_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			mon_sess.run(train_op, feed_dict=feed_dict)

def enqueueInputData(sess, coord, Reader, enqueue_op, queue_input_data , queue_input_target):
	try:	
		while True:
			curr_data, curr_label = Reader.read()
			sess.run(enqueue_op, feed_dict={queue_input_data: curr_data, queue_input_target: curr_label})
	except Exception, e:
		print e
		print("finished enqueueing")
		coord.request_stop(e)

def enqueueInputData_2(sess, coord, Reader, enqueue_op, queue_input_data , queue_input_target):
	while True:
		curr_data, curr_label = Reader.read()
		sess.run(enqueue_op, feed_dict={queue_input_data: curr_data, queue_input_target: curr_label})
	print("finished enqueueing")
	coord.request_stop()

def train():
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	TrainingHand=gzip.open(FLAGS.TrainingData,'rb')
	TrainingReader = RecordReader(TrainingHand)
	
	with tf.Graph().as_default():
		queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT+1) * WIDTH])
		queue_input_label = tf.placeholder(tf.int32, shape=[])
		queue = tf.FIFOQueue(capacity=FLAGS.batch_size*10, dtypes=[dtype, tf.int32], shapes=[[DEPTH * (HEIGHT+1) * WIDTH], []])
		enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
		dequeue_op = queue.dequeue()
		# Get Tensors and labels for Training data.
		data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size*4)
		#data_batch_reshape = tf.transpose(data_batch, [0,2,3,1])

		global_step = tf.Variable(0, trainable=False, name='global_step')

		# Build a Graph that computes the logits predictions from the
		# inference model.
		convnets = Models.ConvNets()
		logits = convnets.Inference(data_batch)

		# Calculate loss.
		loss = convnets.loss(logits, label_batch)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnets.Train(loss, global_step)
		summary = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		sess.run(init)
		
		coord = tf.train.Coordinator()
		enqueue_thread = threading.Thread(target=enqueueInputData, args=[sess, coord, TrainingReader, enqueue_op, queue_input_data, queue_input_label])
		enqueue_thread.isDaemon()
		enqueue_thread.start()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		min_loss = 100
		try:	
			for step in xrange(max_steps):
				start_time = time.time()
				_, loss_value, v_step = sess.run([train_op, loss, global_step])
				curr_batch, curr_label, v_step = sess.run([data_batch, label_batch, global_step])
				duration = time.time() - start_time
				if v_step % 10 == 0:
					print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
					summary_str = sess.run(summary)
					summary_writer.add_summary(summary_str, v_step)
					summary_writer.flush()
				if (v_step) % 100 == 0 or (v_step) == max_steps:
					#Save Model only if loss decreasing
					if loss_value < min_loss:
						checkpoint_file = os.path.join(log_dir, 'model.ckpt')
						saver.save(sess, checkpoint_file, global_step = global_step)
						min_loss = loss_value
					#loss_value = sess.run(loss, feed_dict=feed_dict)
					print 'Step %d Test loss = %.3f (%.3f sec); Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)
		except Exception, e:
			coord.request_stop(e)
		finally:
			sess.run(queue.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)
"""
def continue_train(ModelCKPT):
	with tf.Graph().as_default():
		print "Locating Data File"
		TrainingData = gzip.open(FLAGS.TrainingData,'rb')
		TestingData = gzip.open(FLAGS.TestingData,'rb')
		data_sets_training = Window2Tensor.Data_Reader(TrainingData, batch_size=BATCH_SIZE)
		data_sets_testing = Window2Tensor.Data_Reader(TestingData, batch_size=BATCH_SIZE) 
		print "Training Data @%s; \nTesting Data @%s" % (os.path.abspath(FLAGS.TrainingData), os.path.abspath(FLAGS.TestingData))

		# Get Tensors and labels for Training data.
		#tensors, labels = Models.inputs(FLAGS.data_file)

		global_step = tf.Variable(0, trainable=False, name='global_step')

		# Build a Graph that computes the logits predictions from the
		# inference model.
		tensor_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
		convnets = Models.ConvNets()
		logits = convnets.Inference(tensor_placeholder)

		# Calculate loss.
		loss = convnets.loss(logits, labels_placeholder)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnets.Train(loss, global_step)
		summary = tf.summary.merge_all()

		#init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		sess = tf.Session()
		saver.restore(sess, ModelCKPT)
		#print global_step
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		#sess.run(init)
		
		min_loss = 100
		for step in xrange(max_steps):
			start_time = time.time()
			feed_dict = fill_feed_dict(data_sets_training, tensor_placeholder, labels_placeholder)
			
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			duration = time.time() - start_time
			v_step = sess.run(global_step)    
			if step % 10 == 0:
				print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
				summary_str = sess.run(summary, feed_dict = feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			if (step + 1) % 100 == 0 or (step + 1) == max_steps:
				#Save Model only if loss decreasing
				#print loss_value, min_loss
				if loss_value < min_loss:
					checkpoint_file = os.path.join(log_dir, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step = global_step)
					min_loss = loss_value
				feed_dict = fill_feed_dict(data_sets_testing, tensor_placeholder, labels_placeholder)
				loss_value = sess.run(loss, feed_dict=feed_dict)
				print 'Step %d Test loss = %.3f (%.3f sec). Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)
"""

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--Continue", help="continue training from a checkpoint",
                    type=str)
	args = parser.parse_args()
	if	args.Continue != None:

		if args.Continue.lower() in ['y', 'yes', 't', 'true']:
			return True
		else:
			return False
	else:
		return False

def main(argv=None):  # pylint: disable=unused-argument
	Continue = GetOptions()

	print 'TraingDir is:',FLAGS.train_dir
	if Continue == True:
		ckptfile = FLAGS.checkpoint_dir+'/log/checkpoint'
		f = open(ckptfile,'rb')
		ckpt = f.readline().split(':')[1].strip().strip('"')
		f.close()
		prefix = os.path.abspath(FLAGS.checkpoint_dir+'/log/')
		ckpt = prefix + '/' + ckpt
		print ckpt
		continue_train(ckpt)
	else:
		"""
		cmd = raw_input("Start a New Training?(y/n):")
		if cmd == 'y':
			if tf.gfile.Exists(FLAGS.train_dir):
				tf.gfile.DeleteRecursively(FLAGS.train_dir)
				tf.gfile.MakeDirs(FLAGS.train_dir)
			train()
		else:
			exit()
		"""
		train()

if __name__ == '__main__':
	tf.app.run()
