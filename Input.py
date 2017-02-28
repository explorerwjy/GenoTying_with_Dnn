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
		return tensor_feed, record.pos, [record.label]

"""
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
		print record.pos, record.label
		return tensor,label


def Myloop(coord, Testreader):
	while not coord.should_step():
		try:
			test_tensor, test_label = Testreader.read()
			return test_tensor, test_label
			# Read a example
		except ValueError:
			coord.request_step()

def TestInputQueue():
	"""Train TensorCaller for a number of steps."""
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	with tf.Graph().as_default():

		# Get Tensors and labels for data.
		TestHand=gzip.open(FLAGS.TestingData,'rb')
		Testreader = RecordReader(TestHand)
		#test_tensor, test_label = Testreader.read()
		#test_tensor, test_label = Myloop(coord, Testreader)
		
		# Create a queue, and an op that enqueues examples one at a time in the queue.
		#queue = tf.RandomShuffleQueue(name="TrainingInputQueue", capacity=FLAGS.batch_size*10,min_after_dequeue=FLAGS.batch_size*3, seed=32, dtypes=[dtype, tf.float32], shapes=[[WIDTH,HEIGHT+1,DEPTH], [NUM_CLASSES]])
		queue = tf.FIFOQueue(name="TrainingInputQueue", capacity=FLAGS.batch_size*10, dtypes=[dtype, tf.float32], shapes=[[WIDTH,HEIGHT+1,DEPTH], [1]])
		
		
		enqueue_op = queue.enqueue(Myloop(coord, Testreader))
		#enqueue_op = queue.enqueue([test_tensor, test_label])
		
		
		qr = tf.train.QueueRunner(queue, [enqueue_op] * FLAGS.queueThreads) # Create a queue runner

		tensors, labels = queue.dequeue_many(BATCH_SIZE)
		labels = tf.cast(labels, dtype=tf.int32)

		global_step = tf.Variable(0, trainable=False, name='global_step')


		summary = tf.summary.merge_all()
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		sess = tf.Session()
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
		sess.run(init)

		min_loss = 100	
		coord = tf.train.Coordinator()
		enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

		try:
			for step in xrange(10):
				if coord.should_stop():
					break
				v_tensors, v_labels = sess.run([tensors, labels])
				print v_labels
				print ""

		except Exception, e:
			coord.request_stop(e)
		finally:
			coord.request_stop()
			coord.join(enqueue_threads)
"""

def main():
	TestInputQueue()
	return

if __name__=='__main__':
	main()
