#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Models would like to try on Tensor Variant Caller
#========================================================================================================

from optparse import OptionParser
import tensorflow as tf
from Window2Tensor import *

NUM_CLASSES = 3
Window_Size = (WIDTH * (HEIGHT+1) * 3)



def FullyConnectNN(InputLayer, Hidden1, Hidden2, Hidden3, Hidden4, Hidden5, Hidden6, Hidden7, Hidden8):
	# Hidden 1
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
	# Hidden 4
	with tf.name_scope('hiddel3'):
		weights = tf.Variable(tf.truncated_normal([Hidden3, Hidden4], stddev=1.0 / math.sqrt(float(Hidden3))),
		name='weights')
		biases = tf.Variable(tf.zeros([Hidden4]), name='biases')
		hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)
	# Hidden 5
	with tf.name_scope('hiddel3'):
		weights = tf.Variable(tf.truncated_normal([Hidden4, Hidden5], stddev=1.0 / math.sqrt(float(Hidden4))),
		name='weights')
		biases = tf.Variable(tf.zeros([Hidden5]), name='biases')
		hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases)
	# Hidden 6
	with tf.name_scope('hiddel3'):
		weights = tf.Variable(tf.truncated_normal([Hidden5, Hidden6], stddev=1.0 / math.sqrt(float(Hidden5))),
		name='weights')
		biases = tf.Variable(tf.zeros([Hidden6]), name='biases')
		hidden6 = tf.nn.relu(tf.matmul(hidden5, weights) + biases)
	# Hidden 7
	with tf.name_scope('hiddel3'):
		weights = tf.Variable(tf.truncated_normal([Hidden6, Hidden7], stddev=1.0 / math.sqrt(float(Hidden6))),
		name='weights')
		biases = tf.Variable(tf.zeros([Hidden7]), name='biases')
		hidden7 = tf.nn.relu(tf.matmul(hidden6, weights) + biases)
	# Hidden 8
	with tf.name_scope('hiddel3'):
		weights = tf.Variable(tf.truncated_normal([Hidden7, Hidden8], stddev=1.0 / math.sqrt(float(Hidden7))),
		name='weights')
		biases = tf.Variable(tf.zeros([Hidden8]), name='biases')
		hidden8 = tf.nn.relu(tf.matmul(hidden7, weights) + biases)
	# Output 
	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(tf.truncated_normal([Hidden8, NUM_CLASSES], stddev=1.0 / math.sqrt(float(Hidden8))),
			name='weights')
		biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
		logits = tf.matmul(hidden8, weights) + biases
	return logits

def ConvNet():

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
