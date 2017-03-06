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
import numpy as np
import tensorflow as tf
import Models
from Input import *
import sys
import pysam
sys.stdout = sys.stderr

BATCH_SIZE=FLAGS.batch_size
log_dir = FLAGS.log_dir
max_steps = FLAGS.max_steps


class DataReaderThread(Thread):
	def __init__(self, FileName, sess, coord, queue_input_data, queue_input_label, enqueue_op, subset_i, subset_n):
		self.FileName = FileName
		self.sess = sess
		self.coord = coord
		self.queue_input_data = queue_input_data
		self.queue_input_label = queue_input_label
		self.enqueue_op = enqueue_op
		self.subset_i = subset_i
		self.subset_n = subset_n
		Thread.__init__(self)
	def run(self):
		try:
			tabix_file = pysam.Tabixfile(self.FileName)
			contigs = [contig for contig in tabix_file.contigs]
			contig_subset = contigs[self.subset_i : : self.subset_n]
			print "Lodaing subset %d from %d" % (self.subset_i, self.subset_n)
			Num_of_contigs = len(contig_subset)
			for contig in contig_subset:
				records_iterator = tabix_file.fetch(contig, 0, 10**9, multiple_iterators=True)
				for data, label in self.record_parser(records_iterator):
					#print 'Try to enqueue on process',subset_i
					self.sess.run(self.enqueue_op, feed_dict={self.queue_input_data: data, self.queue_input_label: label})
					#print 'Successful enqueue on process',subset_i

		except Exception, e:
			print e
			print("finished Reading Input Data")
			self.coord.request_stop(e)
	def record_parser(self, records_iterator):
		for line in records_iterator:
			record = window_tensor(line)
			record.encode()
			yield record.res, record.label

def enqueueInputData(sess, coord, Reader, enqueue_op, queue_input_data , queue_input_target):
	try:	
		while True:
			curr_data, curr_label = Reader.read()
			sess.run(enqueue_op, feed_dict={queue_input_data: curr_data, queue_input_target: curr_label})
	except Exception, e:
		print e
		print("finished enqueueing")
		coord.request_stop(e)

def train():
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	#FileName = 'TestInput.txt.gz'
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


		enqueue_thread = Thread(target=enqueueInputData, args=[sess, coord, TrainingReader, enqueue_op, queue_input_data, queue_input_label])
		enqueue_thread.isDaemon()
		enqueue_thread.start()

		#N = 1
		#for i in range(N):
		#	p = DataReaderThread(FileName, sess, coord, queue_input_data, queue_input_label, enqueue_op, i, N)
		#	p.start()

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
					#print "One Batch Reading Costs:",duration
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
