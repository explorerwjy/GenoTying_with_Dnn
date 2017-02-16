#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Models would like to try on Tensor Variant Caller
#========================================================================================================

from optparse import OptionParser
import tensorflow as tf
from Window2Tensor import *
from imput import *

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
NUM_CLASSES = 3
Window_Size = (WIDTH * (HEIGHT+1) * 3)


# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = Input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = Input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = Input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def inputs(data_file):
	tensors, labels = Input.inputs(False, DataFile=data_file, batch_size=FLAGS.batch_size)
	if FLAGS.use_fl16:
		tensors = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return tensors, labels

# This Class is not in used now
class FullyConnectNN():
	def __init__(self):
		pass
	def Inference(self, InputTensor):
		# Input Layer
		with tf.name_scope('hiddel1'):
			weights = tf.Variable(tf.truncated_normal([Window_Size, Hidden1], stddev=1.0 / math.sqrt(float(Window_Size))),
			name='weights')
			biases = tf.Variable(tf.zeros([Hidden1]), name='biases')
			reshaped_InputLayer = tf.reshape(InputLayer, [BATCH_SIZE,Window_Size])
			hidden1 = tf.nn.relu(tf.matmul(reshaped_InputLayer, weights) + biases)
		# Hidden 2
		with tf.name_scope('hiddel2'):
			weights = tf.Variable(tf.truncated_normal([Hidden1, Hidden2], stddev=1.0 / math.sqrt(float(Hidden1))),
			name='weights')
			biases = tf.Variable(tf.zeros([Hidden2]), name='biases')
			hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
		# Hidden 3
		with tf.name_scope('hiddel3'):
			weights = tf.Variable(tf.truncated_normal([Hidden2, Hidden3], stddev=1.0 / math.sqrt(float(Hidden2))),
			name='weights')
			biases = tf.Variable(tf.zeros([Hidden3]), name='biases')
			hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
		# Output 
		with tf.name_scope('softmax_linear'):
			weights = tf.Variable(tf.truncated_normal([Hidden8, NUM_CLASSES], stddev=1.0 / math.sqrt(float(Hidden3))),
				name='weights')
			biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
			logits = tf.matmul(hidden3, weights) + biases
		return logits

class ConvNets():
	def __init__(self):
		pass
	def Inference(self, RawTensor):
		InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT+1, 3]) 

		# conv1
		with tf.variable_scope('conv1') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3,3,3,32], stddev=5e-2, wd=0.0)
			conv = tf.nn.conv2d(InputTensor, kernel, [1,2,2,1], padding='SAME')
			biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(pre_activation, name=scope.name)
			_activation_summary(conv1)
		# pool1
		#pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')
		# norm1
		#norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
		
		# conv2
		with tf.variable_scope('conv2') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3,3,32,64], stddev=5e-2, wd=0.0)
			conv = tf.nn.conv2d(InputTensor, kernel, [1,1,1,1], padding='SAME')
			biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(pre_activation, name=scope.name)
			_activation_summary(conv1)

		# pool1
		pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

		# conv3
		with tf.variable_scope('conv2') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3,3,64,128], stddev=5e-2, wd=0.0)
			conv = tf.nn.conv2d(InputTensor, kernel, [1,1,1,1], padding='SAME')
			biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(pre_activation, name=scope.name)
			_activation_summary(conv1)

		# conv4
		with tf.variable_scope('conv2') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[3,3,128,128], stddev=5e-2, wd=0.0)
			conv = tf.nn.conv2d(InputTensor, kernel, [1,1,1,1], padding='SAME')
			biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(pre_activation, name=scope.name)
			_activation_summary(conv1)


		# conv5
		with tf.variable_scope('conv2') as scope:
			kernel = _variable_with_weight_decay('weights', shape=[5,5,128,256], stddev=5e-2, wd=0.0)
			conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding='SAME')
			biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv2 = tf.nn.relu(pre_activation, name=scope.name)
			_activation_summary(conv2)
		# norm2
		# norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
		# pool2
		pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')
		
		# local6
		with tf.variable_scope('local3') as scope:
			reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
			dim = reshape.get_shape()[1].value
			weights = _variable_with_weight_decay('weights', shape=[dim,384], stddev=0.04, wd=0.004)
			biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
			local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
			_activation_summary(local3)

		# local7
		with tf.variable_scope('local4') as scope:
			weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
			biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
			local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
			_activation_summary(local4)

		# linear layer (WX + b)
		with tf.variable_scope('softmax_linear') as scope:
			weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
			biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
			softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
			_activation_summary(softmax_linear)

		return softmax_linear

	def loss(self, logits, labels):
		labels = tf.cast(labels, tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
		return tf.add_n(tf.get_collection('losses'), name='total_loss')

	def add_loss_summaries(self, total_loss):
		loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
		losses = tf.get_collection('losses')
		loss_averages_op = loss_averages.apply(losses + [total_loss])

		for l in losses + [total_loss]:
			tf.summary.scalar(l.op.name + ' (raw) ', l)
			tf.summary.scalar(l.op.name , loss_averages.average(l))
		return loss_averages_op

	def Train(self, total_loss, global_step):
		num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
		decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
		lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
		tf.summary.scalar('learning_rate', lr)
		loss_averages_op = self.add_loss_summaries(total_loss)

		with tf.control_dependencies([loss_averages_op]):
			opt = tf.train.GradientDescentOptimizer(lr)
			grads = opt.compute_gradients(total_loss)

		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

		for grad, var in grads:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
		variable_averages_op = variable_averages.apply(tf.trainable_variables())

		with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
			train_op = tf.no_op(name='train')
		return train_op


def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var
def _activation_summary(x):
	TOWER_NAME = 'Tower'
	tensor_name = re.sub('%s_[0-9]/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name, shape, stddev, wd):
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def GetOptions():
	parser = OptionParser()
	parser.add_option('-','--',dest = '', metavar = '', help = '')
	(options,args) = parser.parse_args()
	
	return

def main():
	return

if __name__=='__main__':
	main()
