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
		if FLAGS.use_fl16: 
			RawTensor = tf.convert_to_tensor(res, dtype=tf.float16)
		else:
			RawTensor = tf.convert_to_tensor(res, dtype=tf.float32)
		InputTensor = tf.reshape(RawTensor, [WIDTH, HEIGHT+1, 3]) 
		return InputTensor

class RecordReader():
	def __init__(self, hand):
		self.hand = hand
	def read(self):
		line = self.hand.readline()
		if line == '':
			self.hand.seek(0)
			line = self.hand.readline()
		record = window_tensor(self.hand.readline())
		tensor = record.encode()
		#label = tf.one_hot(indices=tf.cast(float(record.label), tf.int32), depth=3)
		label = tf.convert_to_tensor(int(record.label), dtype=tf.float32)
		label = tf.reshape(label, [1])
		pos = tf.convert_to_tensor(record.pos, dtype='tf.string')
		return tensor,pos,label

def enqueue(sess, coord, Testreader, queue):
	try:	
		""" Iterates over our data puts small junks into our queue."""

		#while coord.should_step():
		while True:
			print("starting to write into queue")
			tensor,pos,label = Testreader.read()
			enqueue_op = queue.enqueue(tensor,pos,label)
			pos,_ = sess.run([pos,enqueue_op])
			print("added ",pos,"to the queue")
			print("finished enqueueing")
	except:
		coord.request_stop()

def TestInputQueue():
	"""Train TensorCaller for a number of steps."""
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	TestHand=gzip.open(FLAGS.TestingData,'rb')
	Testreader = RecordReader(TestHand)

	with tf.Graph().as_default():

		# are used to feed data into our queue
		#queue_input_data = tf.placeholder(tf.float32, shape=[20, 4])
		#queue_input_target = tf.placeholder(tf.float32, shape=[20, 3])

		queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32], shapes=[[4], [3]])

		#enqueue_op = queue.enqueue([queue_input_data, queue_pos_data, queue_input_target])
		dequeue_op = queue.dequeue()

		# tensorflow recommendation:
		# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
		data_batch, pos_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
		# use this to shuffle batches:
		# data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)


		init = tf.global_variables_initializer()

		sess = tf.Session()
		
		sess.run(init)

		coord = tf.train.Coordinator()
		enqueue_thread = threading.Thread(target=enqueue, args=[sess, coord, Testreader])
		#enqueue_thread = threading.Thread(target=enqueue, args=[sess, coord, Testreader, enqueue_op])
		enqueue_thread.isDaemon()
		enqueue_thread.start()

		threads = tf.train.start_queue_runners(coord=coord, sess=sess)

		try:
			for step in xrange(10):
				run_options = tf.RunOptions(timeout_in_ms=4000)
				if coord.should_stop():
					break
				curr_data_batch, curr_pos_batch, curr_target_batch = sess.run([data_batch, target_batch], options=run_options)
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
