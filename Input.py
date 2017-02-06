#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Prepare Input Data For Training
#========================================================================================================

from optparse import OptionParser
import os
import Region
import time
import gzip
import tensorflow as tf
import Region

WIDTH = Region.WIDTH
HEIGHT = Region.HEIGHT + 1

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3000


def read_window(filename_queue):
	"""Reads and parses examples from Region Window data files.
	Args:
    inputBuffer: InputBuffer form Input Tensor from.
	Returns:
    An object representing a single example, with the following fields:
      height: number of reads in the result 
      width: number of bases in the result 
      depth: number of layers in the result 
      key: a scalar string Tensor describing the chrom:pos for this example.
      label: an int32 Tensor with the label in the range 0,1,2
      uint8image: a [height, width, depth] uint8 Tensor with the read data
	"""
	"""
	class window_tensor():
		def __init__(self,line):
			self.label = line[0]
			self.Alignment = line[ 13 : 13 + WIDTH * (HEIGHT+1) ]
			self.Qual = line[ 13 + WIDTH * (HEIGHT+1) : 13 + WIDTH * (HEIGHT+1)*2]
			self.Strand = line[13 + WIDTH * (HEIGHT+1)*2 : 13 + WIDTH * (HEIGHT+1)*3]

		def encode(self):
			# This func encode elements in window tensor into tf.float32
			return map(float, list(self.Alignment)) + map(lambda x:qual2code(x), list(self.Qual)) + map(float, list(self.Strand))
	tmp = window_tensor(inputBuffer.strip())
	one_tensor = tmp.encode()
	one_label = tmp.label
	"""
	class Record(object):
		pass
	result = Record()
	
	reader = tf.TextLineReader()
	result.key, value = reader.read(filename_queue)

	# Convert from a string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)
	tensor_bytes = (HEIGHT) * WIDTH * 3
	print record_bytes
	# The first bytes represent the label, which we convert from uint8->int32.
	result.label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
	result.pos = tf.cast(tf.slice(record_bytes, [1], [13]), tf.int32)
	# The remaining bytes after the label represent the image, which we reshape
	# from [depth * height * width] to [depth, height, width].
	print tensor_bytes
	depth_major = tf.reshape(
    	tf.slice(record_bytes, [13], [13 + tensor_bytes]), [3, HEIGHT, WIDTH])
	# Convert from [depth, height, width] to [height, width, depth].
	result.tensor = tf.transpose(depth_major, [1, 2, 0])

	return result

#  Construct a queued batch of images and labels.
def Generate_Tensor_and_label_batch(tensor, label, min_queue_examples, batch_size, shuffle):
	"""
	Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
	Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
	"""
	num_preprocess_threads = 16
	if shuffle:
		Tensors, Labels_batch = tf.train.batch([tensor, label], batch_size = batch_size, num_threads = num_preprocess_threads, capacity = min_queue_examples + 3*batch_size)
	else:
		Tensors, Labels_batch = tf.train.batch([tensor, label], batch_size = batch_size, num_threads = num_preprocess_threads, capacity = min_queue_examples + 3*batch_size)
	# How this work? Display the training Tensor as image?
	tf.image_summary('tensors',Tensors)
	return Tensors, tf.reshape(Label_batch, [batch_size])

def inputs(DataFile, batch_size):
	"""Construct input for CIFAR evaluation using the Reader ops.
	Args:
	DataFile: Input File contains window tensor.
    batch_size: Number of images per batch.
	Returns:
    tensors: Tensors. 4D tensor of [batch_size, WIDTH, HEIGHT, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
	"""
	# Create a queue that produces the buffer to read.
	filename_queue = tf.train.string_input_producer([DataFile])

	# Read examples from buffers in the buffer queue.
	read_input = read_window(filename_queue)
	reshaped_image = tf.cast(read_input.tensor, tf.float32)

	# Set the shapes of tensors.
	float_image.set_shape([HEIGHT, WIDTH, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

	# Generate a batch of images and labels by building up a queue of examples.
	return Generate_Tensor_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

class Data():
	def __init__(self,TrainingHand,ValidationHand,TestingHand):
		self.TrainingHand = TrainingHand
		self.TrainingBatch = BatchSize
		self.ValidationHand = ValidationHand
		self.TestingHand = TestingHand
	def ReadingTraining(self):
		# Args:    file hand of Training windows, get [batch_size] Lines of records. 
		# Returns: 4D tensor of [batch_size, height, width, depth] size
		#		   1D tensor of [batch_size] size.
		self.Training = []
		self.TrainingLabels = []
		i = 0
		while i < self.TrainingBatch:
			l = TrainingHand.readline()
			if l == '':
				self.handle.seek(0)
				continue
			tensor, label = Record(l)
			self.Training.append(tensor)
			self.label.append(label)
			i += 1
		return self.Training, self.TrainingLabels
	def ReadingValidation(self):
		self.Validation = []
		self.ValidationLabels = []
		for l in self.ValidationHand:
			tensor, label = Record(l)
			self.Validation.append(tensor)
			self.ValidationLabels.append(labels)
		return self.Validation, self.ValidationLabels
	def ReadingTesting(self):
		self.Testing = []
		self.TestingLabels = []
		for l in self.TestingHand:
			tensor, label = Record(l)
			self.Testing.append(tensor)
			self.TestingLabels.append(labels)
		return self.Testing, self.TestingLabels
def Inputs_training(Training_fname, batch_size):
	
	
	return
def Inputs_validation_testing():

	return
def GetOptions():
	parser = OptionParser()
	parser.add_option('-','--',dest = '', metavar = '', help = '')
	(options,args) = parser.parse_args()
	
	return

def main():

	return

if __name__=='__main__':
	main()
