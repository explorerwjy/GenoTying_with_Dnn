#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Prepare Input Data For Training
#========================================================================================================

from optparse import OptionParser
import os
import Region
import time
import gzip
import threading
from threading import Thread
import Queue
import numpy as np
import tensorflow as tf


# Basic model parameters.
WIDTH = Region.WIDTH
HEIGHT = Region.HEIGHT
DEPTH = Region.DEPTH
Window_Size = (WIDTH * (HEIGHT+1) * DEPTH)

NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 6400
LEARNING_RATE_DECAY_STEP = 1000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# Global constants describing the data set & Model.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', './tmp/TensorCaller_eval',
		"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('train_dir', './tmp/TensorCaller_train',
		"""Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', './tmp/TensorCaller_train/log',
		"""Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
		"""How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
		"""Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
		"""Number of WindowTensor to process in a batch.""")
tf.app.flags.DEFINE_string('TrainingData', './Training.windows.txt.gz',
		"""Path to the Training Data.""")
tf.app.flags.DEFINE_string('ValidationData', '',
		"""Path to the Validation Data.""")
tf.app.flags.DEFINE_string('TestingData', 'Testing.windows.txt.gz',
		"""Path to the Testing Data.""")
tf.app.flags.DEFINE_boolean('use_fl16', True,
		"""Train the model using fp16.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
		"""Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
		"""Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('queueThreads', 4,
		"""Number of threads used to read data""")

# ==========================================================================
def base2code(base):
	try:
		#return tf.cast(BASE[base],tf.float32)
		#return float(BASE[base])
		return float(base)
	except KeyError:
		print "KeyError of base code. Unexpected base appear. |%s|" % base
		exit()
def qual2code(ch):
	phred = (float(ord(ch) - 33) / 60) - 0.5
	#return tf.cast((math.pow(10, -(phred/10))),tf.float32)
	#return float(math.pow(10, -(phred/10)))
	return phred 
def strand2code(ch):
	return float(ch)
# ==========================================================================

class window_tensor():
	def __init__(self,line):
		self.chrom, self.start, self.end, self.ref, self.alt, self.label, self.window = line.strip().split('\t')
		self.Alignment = self.window[ 0 : WIDTH * (HEIGHT+1) ]
		self.Qual = self.window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
		self.Strand = self.window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
		self.pos = self.chrom+':'+self.start

	def encode(self):
		# This func encode,norm elements and form into tensor 
		res = [ (float(base)/6 - 0.5) for base in list(self.Alignment)] + [ qual2code(x) for x in list(self.Qual)] + [ float(x)/2-0.5 for x in list(self.Strand)]
		return np.array(res)

class RecordReader():
	def __init__(self, hand):
		self.hand = hand
	def read(self):
		line = self.hand.readline()
		if line == '':
			self.hand.seek(0)
			line = self.hand.readline()
		record = window_tensor(self.hand.readline())
		flat_alignment = record.encode()
		tensor_feed = flat_alignment.reshape(WIDTH,HEIGHT+1,DEPTH)
		#label = tf.one_hot(indices=tf.cast(float(record.label), tf.int32), depth=3)
		#label = tf.convert_to_tensor(int(record.label), dtype=tf.float32)
		#label = tf.reshape(label, [1])
		#pos = tf.convert_to_tensor(record.pos, dtype='tf.string')
		#return tensor,pos,label
		#print tensor_feed
		return tensor_feed, record.pos, [record.label]

class Worker(Thread):
	"""Thread executing tasks from a given tasks queue"""
	def __init__(self, tasks):
		Thread.__init__(self)
		self.tasks = tasks
		self.daemon = True
		self.start()
	def run(self):
		while True:
			func, args, kargs = self.tasks.get()
			try:
				func(*args, **kargs)
			except Exception, e:
				print e
			finally:
				self.tasks.task_done()

class ThreadPool():
	"""Pool of threads consuming tasks from a queue"""
	def __init__(self, num_threads):
		self.tasks = Queue(num_threads)
		for _ in range(num_threads): Worker(self.tasks)

	def add_task(self, func, *args, **kargs):
		"""Add a task to the queue"""
		self.tasks.put((func, args, kargs))

	def wait_completion(self):
		"""Wait for completion of all the tasks in the queue"""
		self.tasks.join()

def DecodeAndEnQueue(sess, coord, enqueue_op, queue_input_data, queue_input_pos, queue_input_target, line):
	record = window_tensor(line)
	flat_alignment = record.encode()
	tensor_feed = flat_alignment.reshape(WIDTH,HEIGHT+1,DEPTH)

	return tensor_feed, record.pos, [record.label]
	sess.run(enqueue_op, feed_dict={queue_input_data: curr_data, queue_input_pos: curr_pos, queue_input_target: curr_label})
	print "added ",curr_pos,"to the queue"


# One Main Thread Readfile, other threads process it.
def ReadingData_2(sess, coord, num_threads, enqueue_op, queue_input_data, queue_input_pos, queue_input_target):
	TestHand=gzip.open(FLAGS.TestingData,'rb')
	try:
		while True:
			for i in range(num_threads):
				line = TestHand.readline()
				if line == '':
					self.hand.seek(0)
					line = TestHand.readline()
				enqueue_thread = threading.Thread(target=DecodeAndEnQueue, args=[sess, coord, enqueue_op, queue_input_data,queue_input_pos, queue_input_target, line])
				enqueue_thread.isDaemon()
				enqueue_thread.start()
	except:
		print("finished enqueueing")
		TestHand.close()
		coord.request_stop()
			
def ReadingData(sess, coord, num_threads, enqueue_op, queue_input_data, queue_input_pos, queue_input_target):
	TestHand=gzip.open(FLAGS.TestingData,'rb')
	pool = ThreadPool(20)
	try:
		while True:
			for i in range(num_threads):
				line = TestHand.readline()
				if line == '':
					self.hand.seek(0)
					line = TestHand.readline()

				pool.add_task(DecodeAndEnQueue, args=[sess, coord, enqueue_op, queue_input_data,queue_input_pos, queue_input_target, line])

				#enqueue_thread = threading.Thread(target=DecodeAndEnQueue, args=[sess, coord, enqueue_op, queue_input_data,queue_input_pos, queue_input_target, line])
				#enqueue_thread.isDaemon()
				#enqueue_thread.start()

		pool.wait_completion()
	except:
		print("finished enqueueing")
		TestHand.close()
		coord.request_stop()



def TestInputQueue():
	"""Train TensorCaller for a number of steps."""
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	with tf.Graph().as_default():
		queue_input_data = tf.placeholder(tf.float32, shape=[101,101,3])
		queue_input_pos = tf.placeholder(tf.string, shape=[])
		queue_input_target = tf.placeholder(tf.float32, shape=[1])
		queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.string, tf.float32], shapes=[[WIDTH,HEIGHT+1,DEPTH], [], [1]])

		enqueue_op = queue.enqueue([queue_input_data, queue_input_pos, queue_input_target])
		dequeue_op = queue.dequeue()

		# tensorflow recommendation:
		# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
		data_batch, pos_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		coord = tf.train.Coordinator()
		num_threads = FLAGS.queueThreads
		InputQueueRunner = threading.Thread(target=ReadingData, args= [sess, coord, num_threads, enqueue_op, queue_input_data, queue_input_pos, queue_input_target])
		InputQueueRunner.isDaemon()
		InputQueueRunner.start()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		try:
			for step in xrange(3):
				run_options = tf.RunOptions(timeout_in_ms=4000)
				if coord.should_stop():
					break
				curr_data_batch, curr_pos_batch, curr_target_batch = sess.run([data_batch, pos_batch, target_batch], options=run_options)
				print curr_data_batch
				print curr_target_batch

		except Exception, e:
			coord.request_stop(e)
		finally:
			sess.run(queue.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)


def main():
	TestInputQueue()
	return

if __name__=='__main__':
	main()
