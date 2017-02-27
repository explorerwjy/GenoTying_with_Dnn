#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Training The ConvNet for Tensor Caller
#========================================================================================================

import argparse
from datetime import datetime
import time
import os 
import tensorflow as tf
import Models
from Input import *
#import Window2Tensor
import sys
sys.stdout = sys.stderr

BATCH_SIZE=FLAGS.batch_size
log_dir = FLAGS.log_dir
max_steps = FLAGS.max_steps


def GetInputs_2():
	print "Locating Data File"
	TrainingData = gzip.open(FLAGS.TrainingData,'rb')
	TestingData = gzip.open(FLAGS.TestingData,'rb')
	data_sets_training = Window2Tensor.Data_Reader(TrainingData, batch_size=BATCH_SIZE)
	data_sets_testing = Window2Tensor.Data_Reader(TestingData, batch_size=BATCH_SIZE) 
	print "Training Data @%s; \nTesting Data @%s" % (os.path.abspath(FLAGS.TrainingData), os.path.abspath(FLAGS.TestingData))

def GetInputs():
	tensors, label = queue.dequeue_many(batch_size)

def train():
	"""Train TensorCaller for a number of steps."""
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	with tf.Graph().as_default():

		# Get Tensors and labels for Training data.
		TrainHand=gzip.open(FLAGS.TrainingData,'rb')
		TestHand=gzip.open(FLAGS.TestingData,'rb')
		Trainreader = RecordReader(TrainHand)
		Testreader = RecordReader(TestHand)
		train_tensor, train_label = Trainreader.read()
		test_tensor, test_label = Testreader.read()

		# Create a queue, and an op that enqueues examples one at a time in the queue.
		queue = tf.RandomShuffleQueue(name="TrainingInputQueue", capacity=FLAGS.batch_size*10,min_after_dequeue=FLAGS.batch_size*3, seed=32, dtypes=[dtype, tf.float32], shapes=[[WIDTH,HEIGHT+1,DEPTH], [NUM_CLASSES]])
		enqueue_op = queue.enqueue([train_tensor, train_label])
		qr = tf.train.QueueRunner(queue, [enqueue_op] * FLAGS.queueThreads) # Create a queue runner

		tensors, labels = queue.dequeue_many(BATCH_SIZE)
		labels = tf.cast(labels, dtype=dtype)
		print tensors, labels
	
		global_step = tf.Variable(0, trainable=False, name='global_step')

		# Build a Graph that computes the logits predictions from the
		# inference model.
		convnets = Models.ConvNets()
		logits = convnets.Inference(tensors)
		print 'logits',logits
		print 'lables',labels
		
		# Calculate loss.
		loss = convnets.loss(logits, labels)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = convnets.Train(loss, global_step)
		summary = tf.summary.merge_all()
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		sess.run(init)
		
		min_loss = 100	
		coord = tf.train.Coordinator()
		enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

		try:
			for step in xrange(max_steps):
				start_time = time.time()
				if coord.should_stop():
					break
				_, loss_value, v_step = sess.run([train_op, loss, global_step])
				duration = time.time() - start_time
				if v_step % 10 == 0:
					print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
					summary_str = sess.run(summary)
					summary_writer.add_summary(summary_str, v_step)
					summary_writer.flush()

				if (v_step) % 100 == 0 or (v_step) == max_steps:
					if loss_value < min_loss:
						checkpoint_file = os.path.join(log_dir, 'model.ckpt')
						saver.save(sess, checkpoint_file, global_step = global_step)
						min_loss = loss_value
					loss_value = sess.run(loss)
					print 'Saved loss = %.3f' % (min_loss)
					#print 'Step %d Test loss = %.3f (%.3f sec); Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)

		except Exception, e:
			coord.request_stop(e)
		finally:
			coord.request_stop()
			coord.join(enqueue_threads)


def continue_train(ModelCKPT):
	"""Train TensorCaller for a number of steps."""
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
		ckptfile = FLAGS.log_dir+'/log/checkpoint'
		f = open(ckptfile,'rb')
		ckpt = f.readline().split(':')[1].strip().strip('"')
		f.close()
		prefix = os.path.abspath(FLAGS.log_dir)
		ckpt = prefix + '/' + ckpt
		print ckpt
		continue_train(ckpt)
	else:
		train()

if __name__ == '__main__':
	main()
	#tf.app.run()
