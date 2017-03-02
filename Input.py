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
import multiprocessing
import tensorflow as tf

# Basic model parameters.
WIDTH = Region.WIDTH
HEIGHT = Region.HEIGHT
DEPTH = 3
Window_Size = (WIDTH * (HEIGHT+1) * 3)

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

tf.app.flags.DEFINE_string('train_dir', './tmp/TensorCaller_train_2',
                           """Directory where to write event logs """
                           """and checkpoint.""")
	
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/TensorCaller_train_2',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_string('log_dir', './tmp/TensorCaller_train_2/log',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of WindowTensor to process in a batch.""")

#tf.app.flags.DEFINE_integer('test_batch_size', 64,
#                            """Number of WindowTensor to process in a batch.""")

tf.app.flags.DEFINE_string('TrainingData', 'Training.windows.txt.gz',
                           """Path to the Training Data.""")

tf.app.flags.DEFINE_string('ValidationData', './windows_validation.txt.gz',
                           """Path to the Validation Data.""")

tf.app.flags.DEFINE_string('TestingData', 'Testing.windows.txt.gz',
                           """Path to the Testing Data.""")

tf.app.flags.DEFINE_boolean('use_fl16', False,
                            """Train the model using fp16.""")


tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('numOfDecodingThreads', 4,
                            """Whether to log device placement.""")

npdtype = np.float16 if FLAGS.use_fl16 else np.float32
# ARG: batch_size: The batch size will be baked into both placeholders.
# Return: Tensors placeholder, Labels placeholder.

class window_tensor():
	def __init__(self,line):
		self.chrom, self.start, self.end, self.ref, self.alt, self.label, self.window = line.strip().split('\t')
		self.Alignment = self.window[ 0 : WIDTH * (HEIGHT+1) ]
		self.Qual = self.window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
		self.Strand = self.window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
	def encode(self):
		# This func encode elements in window tensor into tf.float32
		
		#print len(self.Alignment), len(self.Alignment[0])
		#print len(self.Qual), len(self.Qual[0])
		#print len(self.Strand), len(self.Strand[0])
		#exit()
		p1 = np.fromiter(self.Alignment, dtype = npdtype)
		p2 = np.array(map(lambda x: qual2code(x), self.Qual), dtype = npdtype)
		p3 = np.fromiter(self.Strand, dtype = npdtype)
		self.res = np.concatenate([p1, p2, p3])
		
		#res = [ (base2code(base)/6 - 0.5) for base in self.Alignment] + [ qual2code(x) for x in self.Qual] + [float(x)/2-0.5 for x in self.Strand] 
		#res = map(lambda x:(float(x)/6 - 0.5), list(self.Alignment)) + map(lambda x: qual2code(x), list(self.Qual)) + map(lambda x:float(x)/2-0.5, list(self.Strand)) 
		#res = map(lambda x: qual2code(x), self.Qual) 
		#res = map(lambda x:(float(x)/6 - 0.5), list(self.Alignment))
		#res = map(lambda x:(float(x)/6 - 0.5), list(self.Alignment)) + map(lambda x: qual2code(x), list(self.Qual)) + map(lambda x:float(x)/2-0.5, list(self.Strand)) 
		#print len([float(x)/2-0.5 for x in list(self.Strand)]), len(list(self.Strand))
		#return np.array(res)
"""
class window_tensor():
	def __init__(self, line):
		self.chrom, self.start, self.end, self.ref, self.alt, self.label, self.window = line.strip().split('\t')
		self.res = np.fromiter(self.window[ 0 : WIDTH * (HEIGHT+1) ], dtype = npdtype) + np.array(map(lambda x: qual2code(x), self.window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]), dtype = npdtype) + np.fromiter(self.window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3], dtype = npdtype)
	def encode(self):
		return self.res
"""
class RecordReader():
	def __init__(self, handle):
		self.hand = handle
	def read(self):
		line = self.hand.readline()
		if line == '':
			self.hand.seek(0)
			line = self.hand.readline()
		record = window_tensor(line)
		record.encode()
		#return flat_alignment, record.label
		#print len(record.res)
		return record.res, record.label
	def read_without_processing(self):
		line = self.hand.readline()
		if line == '':
			self.hand.seek(0)
			line = self.hand.readline()
		record = window_tensor(self.hand.readline())
# ==========================================================================
#@vectorize(["float32(chr)"], target='gpu')
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

def TestReadingTime():
	Hand = gzip.open(FLAGS.TrainingData,'rb')
	Reader = RecordReader(Hand)
	CompareSteps = 128
	print 'Reading with decoding'
	count = 0
	s_time = time.time()
	while count < CompareSteps:
		Reader.read()
		count += 1
	print "With Decoding: ",time.time()-s_time
	print "\nReading without Decoding..."
	s_time = time.time()
	count = 0
	while count < CompareSteps:
		Reader.read_without_processing()
		count += 1
	print "Without Decoding: ",time.time()-s_time




def main():
	TestReadingTime()
	return

if __name__=='__main__':
    main()
