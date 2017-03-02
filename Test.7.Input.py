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
import multiprocessing
from multiprocessing import Process, Pool
from Queue import Queue
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

DataDecodingQueue = Queue(BATCH_SIZE*10000)
DecodeBatch = 10000

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
			record_batch = []
			while True:
				line = self.hand.readline()
				if line == '':
					self.hand.seek(0)
					line = self.hand.readline()
				record = window_tensor(line)
				record_batch.append(record)
				if len(record_batch) == DecodeBatch:
					DataDecodingQueue.put(record_batch)
					record_batch = []
		except Exception, e:
			print e
			print("finished Reading Input Data")
			self.coord.request_stop(e)

class InputDataDecoder(Process):
	def __init__(self, sess, coord, enqueue_op, queue_input_data , queue_input_label, _id):
		self.sess = sess
		self._id = _id
		self.coord = coord
		self.enqueue_op = enqueue_op
		self.queue_input_data = queue_input_data
		self.queue_input_label = queue_input_label
		Process.__init__(self)
	def run(self):
		print "Starting Decoder", self._id
		global DataDecodingQueue
		try:
			while True:
				record_batch = DataDecodingQueue.get()
				#print record.chrom + ':' + record.start
				for record in record_batch:
					record.encode()
					self.sess.run(self.enqueue_op, feed_dict={self.queue_input_data: record.res , self.queue_input_label: record.label})
		except Exception, e:
			print e
			print("finished enqueueing")
			self.coord.request_stop(e)

def load_samples(FileName, sess, coord, queue_input_data, queue_input_label, enqueue_op, subset_i, subset_n):
	try:
		tabix_file = pysam.Tabixfile(FileName)
		contigs = [contig for contig in tabix_file.contigs]
		contig_subset = contigs[subset_i : : subset_n]
		print "Lodaing subset %d from %d" % (subset_i, subset_n)
		Num_of_contigs = len(contig_subset)
		for contig in contig_subset:
			records_iterator = tabix_file.fetch(contig, 0, 10**9, multiple_iterators=True)
			for data, label in record_parser(records_iterator):
				print 'Try to enqueue on process',subset_i
				sess.run(enqueue_op, feed_dict={queue_input_data: data, queue_input_label: label})
				print 'Successful enqueue on process',subset_i

	except Exception, e:
		print e
		print("finished Reading Input Data")
		coord.request_stop(e)

class ProcessWorker(Process):
	def __init__(self, queue):
		self.queue = queue
		super(ProcessWorker, self).__init__()
	def run(self):
		for data in iter( self.queue.get ):


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
			DecoderQueue = Queue(1000000)
			procs = []
			for contig in contig_subset:
				
				records_iterator = tabix_file.fetch(contig, 0, 10**9, multiple_iterators=True) # A records_iterator is one contig of sample lines

				while 

		except Exception, e:
			print e
			print("finished Reading Input Data")
			self.coord.request_stop(e)
	def record_parser(self, records_iterator):
		for line in records_iterator:
			record = window_tensor(line)
			record.encode()
			yield record.res, record.label

def ProcessWorker():


def train():
	# Try multiProcess
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	
	print "Start Building Graph"
	with tf.Graph().as_default():
		print "Define Training Data FIFOQueue"
		queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT+1) * WIDTH])
		queue_input_label = tf.placeholder(tf.int32, shape=[])
		queue = tf.FIFOQueue(capacity=FLAGS.batch_size*10000, dtypes=[dtype, tf.int32], shapes=[[DEPTH * (HEIGHT+1) * WIDTH], []])
		enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
		dequeue_op = queue.dequeue()
		# Get Tensors and labels for Training data.
		data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size*10000)

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		
		coord = tf.train.Coordinator()

	
		print "Lunch Process reading & decoding data from file."
		FileName = 'TestInput.txt.gz'
		N = 2
		for i in range(N):
			p = DataReaderThread(FileName, sess, coord, queue_input_data, queue_input_label, enqueue_op, i, N)
			p.start()

		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
		print "Start Runing"
		min_loss = 100
		try:	
			for step in xrange(max_steps):
				print '='*50+'\n',step
				start_time = time.time()
				curr_data, curr_labels, NumQ = sess.run([data_batch, label_batch, queue.size()])
				print "1 Batch Time Costs: %.3f, 1 Batch size: %d" % ((time.time() - start_time), len(curr_labels))
				print "Num Ele in Queue:",NumQ
		except Exception, e:
			coord.request_stop(e)
		finally:
			sess.run(queue.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)

def train_2():
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	
	print "Start Building Graph"
	with tf.Graph().as_default():
		print "Define Training Data FIFOQueue"
		queue_input_data = tf.placeholder(dtype, shape=[DEPTH * (HEIGHT+1) * WIDTH])
		queue_input_label = tf.placeholder(tf.int32, shape=[])
		queue = tf.FIFOQueue(capacity=FLAGS.batch_size*10000, dtypes=[dtype, tf.int32], shapes=[[DEPTH * (HEIGHT+1) * WIDTH], []])
		enqueue_op = queue.enqueue([queue_input_data, queue_input_label])
		dequeue_op = queue.dequeue()
		# Get Tensors and labels for Training data.
		data_batch, label_batch = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size*10000)

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		
		coord = tf.train.Coordinator()

	
		print "Lunch Threads reading & decoding data from file."
		Producer = InputDataProducer(sess, coord, FLAGS.TrainingData)
		Producer.isDaemon()
		Producer.start()
		for i in range(4):
			Consumer = InputDataDecoder(sess, coord, enqueue_op, queue_input_data , queue_input_label, i)
			#Consumer.isDaemon()
			Consumer.start()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
		
		print "Start Runing"
		min_loss = 100
		try:	
			for step in xrange(max_steps):
				start_time = time.time()
				curr_data, curr_labels, NumQ = sess.run([data_batch, label_batch, queue.size()])
				print "1 Batch Time Costs: %.3f, 1 Batch size: %d" % ((time.time() - start_time), len(curr_labels))
				print "Num Ele in Queue:",NumQ
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
