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
import tensorflow as tf
import Models
from Input import *
#import Window2Tensor
import sys
sys.stdout = sys.stderr

BATCH_SIZE=FLAGS.batch_size
log_dir = FLAGS.log_dir
max_steps = FLAGS.max_steps

def enqueue(sess, coord, Reader, enqueue_op, queue_input_data, queue_input_pos, queue_input_target):
	try:	
		while True:
			#print("starting to write into queue")
			curr_data, curr_pos, curr_label = Reader.read()
			#print queue_input_data, queue_input_pos, queue_input_target
			sess.run(enqueue_op, feed_dict={queue_input_data: curr_data, queue_input_pos: curr_pos, queue_input_target: curr_label})
			#print "added ",curr_pos,"to the queue" 
		#print("finished enqueueing")
	except:
		print("finished enqueueing")
		coord.request_stop()

def train():
	"""Train TensorCaller for a number of steps."""
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	TrainingHand=gzip.open(FLAGS.TrainingData,'rb')
	TrainingReader = RecordReader(TrainingHand)

	with tf.Graph().as_default():

		queue_input_data = tf.placeholder(dtype, shape=[WIDTH,HEIGHT+1,DEPTH])
		queue_input_pos = tf.placeholder(tf.string, shape=[])
		queue_input_label = tf.placeholder(tf.int32, shape=[])

		queue = tf.FIFOQueue(capacity=FLAGS.batch_size*10, dtypes=[dtype, tf.string, tf.int32], shapes=[[WIDTH,HEIGHT+1,DEPTH], [], []])

		enqueue_op = queue.enqueue([queue_input_data, queue_input_pos, queue_input_label])
		dequeue_op = queue.dequeue()
		data_batch, pos_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size*4)
	
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
		
		min_loss = 100	
		coord = tf.train.Coordinator()
		enqueue_thread = threading.Thread(target=enqueue, args=[sess, coord, TrainingReader, enqueue_op, queue_input_data,queue_input_pos, queue_input_label])
		enqueue_thread.isDaemon()
		enqueue_thread.start()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		try:
			for step in xrange(max_steps):
				start_time = time.time()
				if coord.should_stop():
					break
				_, loss_value, v_step = sess.run([train_op, loss, global_step])
				duration = time.time() - start_time
				if v_step % 10 == 0:
					Batch_pos, Batch_label = sess.run([pos_batch, target_batch])
					print Batch_pos, Batch_label
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
			sess.run(queue.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)


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
