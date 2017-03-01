#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Training The ConvNet for Tensor Caller
#========================================================================================================

import argparse
from datetime import datetime
import time
import os 
from threading import Thread
from Queue import Queue
import numpy as np
import tensorflow as tf
import Models
from Input import *
import sys
sys.stdout = sys.stderr

BATCH_SIZE=FLAGS.batch_size
log_dir = FLAGS.log_dir
max_steps = FLAGS.max_steps

DataDecodingQueue = Queue(BATCH_SIZE*10)

class InputDataProducer(Thread):
	def __init__(self, sess, coord, FileName):
		self.sess = sess
		self.coord = coord
		self.hand = gzip.open(FileName,'rb')
		Thread.__init__(self)
	def run(self):
		print 'Starting Data Reader'
		global DataDecodingQueue
		try:
			while True:
				line = self.hand.readline()
				if line == '':
					self.hand.seek(0)
					line = self.hand.readline()
				record = window_tensor(line)
				DataDecodingQueue.put(record)
		except Exception, e:
			print e
			print("finished Reading Input Data")
			self.coord.request_stop(e)

class InputDataDecoder(Thread):
	def __init__(self, sess, coord, enqueue_op, queue_input_data , queue_input_label, _id):
		self.sess = sess
		self._id = _id
		self.coord = coord
		self.enqueue_op = enqueue_op
		self.queue_input_data = queue_input_data
		self.queue_input_label = queue_input_label
		Thread.__init__(self)
	def run(self):
		print "Starting Decoder", self._id
		global DataDecodingQueue
		try:
			while True:
				record = DataDecodingQueue.get()
				#print record.chrom + ':' + record.start
				record.encode()
				self.sess.run(self.enqueue_op, feed_dict={self.queue_input_data: record.res , self.queue_input_label: record.label})
		except Exception, e:
			print e
			print("finished enqueueing")
			self.coord.request_stop(e)


def train():
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	
	print "Start Building Graph"
	with tf.Graph().as_default():
		print "Define Training Data FIFOQueue"
		queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT+1) * WIDTH])
		queue_input_label = tf.placeholder(tf.int32, shape=[])
		queue = tf.FIFOQueue(capacity=FLAGS.batch_size*10, dtypes=[dtype, tf.int32], shapes=[[DEPTH * (HEIGHT+1) * WIDTH], []])
		enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
		dequeue_op = queue.dequeue()
		# Get Tensors and labels for Training data.
		data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size*4)
		#data_batch_reshape = tf.transpose(data_batch, [0,2,3,1])

		#global_step = tf.Variable(0, trainable=False, name='global_step')

		# Build a Graph that computes the logits predictions from the
		# inference model.
		#convnets = Models.ConvNets()
		#logits = convnets.Inference(data_batch)

		# Calculate loss.
		#loss = convnets.loss(logits, label_batch)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		#train_op = convnets.Train(loss, global_step)
		#summary = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		#saver = tf.train.Saver()
		sess = tf.Session()
		#summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		sess.run(init)
		
		coord = tf.train.Coordinator()

	
		print "Lunch Threads reading & decoding data from file."
		# Manipulate Data Streaming
		#enqueue_thread = threading.Thread(target=enqueueInputData, args=[sess, coord, TrainingReader, enqueue_op, queue_input_data, queue_input_label])
		#enqueue_thread.isDaemon()
		#enqueue_thread.start()
		Producer = InputDataProducer(sess, coord, FLAGS.TrainingData)
		Producer.isDaemon()
		Producer.start()
		for i in range(FLAGS.numOfDecodingThreads):
			Consumer = InputDataDecoder(sess, coord, enqueue_op, queue_input_data , queue_input_label, i)
			Consumer.isDaemon()
			Consumer.start()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
		
		print "Start Runing"
		min_loss = 100
		try:	
			for step in xrange(max_steps):
						
				start_time = time.time()

				curr_data, curr_labels = sess.run([data_batch, label_batch])
				
				print "1 Batch Time Costs:", time.time() - start_time
				#_, loss_value, v_step = sess.run([train_op, loss, global_step])

				#for label in curr_labels:
				#	print label,
				#print ''
				
				#if v_step % 10 == 0:
					#print 'Step %d Training loss = %.3f (%.3f sec)' % (v_step, loss_value, duration)
					#summary_str = sess.run(summary)
					#summary_writer.add_summary(summary_str, v_step)
					#summary_writer.flush()

				#if (v_step) % 100 == 0 or (v_step) == max_steps:
					#Save Model only if loss decreasing
					#if loss_value < min_loss:
					#	checkpoint_file = os.path.join(log_dir, 'model.ckpt')
					#	saver.save(sess, checkpoint_file, global_step = global_step)
					#	min_loss = loss_value
					#loss_value = sess.run(loss, feed_dict=feed_dict)
					#print 'Step %d Test loss = %.3f (%.3f sec); Saved loss = %.3f' % (v_step, loss_value, duration, min_loss)
		except Exception, e:
			coord.request_stop(e)
		finally:
			sess.run(queue.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)


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
		train()

if __name__ == '__main__':
	tf.app.run()
