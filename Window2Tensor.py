#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Read window from file and transform them into tensor
#========================================================================================================

from optparse import OptionParser
import tensorflow as tf
from Region import *
import gzip
import time
import math

BATCH_SIZE = 32

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
	phred = float(ord(ch) - 33)
	#return tf.cast((math.pow(10, -(phred/10))),tf.float32)
	#return float(math.pow(10, -(phred/10)))
	return phred 
def strand2code(ch):
	return float(ch)
# ==========================================================================


class window_tensor():
	def __init__(self,line):
		self.label = line[0]
		self.Alignment = line[ 13 : 13 + WIDTH * (HEIGHT+1) ]
		self.Qual = line[ 13 + WIDTH * (HEIGHT+1) : 13 + WIDTH * (HEIGHT+1)*2]
		self.Strand = line[13 + WIDTH * (HEIGHT+1)*2 : 13 + WIDTH * (HEIGHT+1)*3]

	def encode(self):
		# This func encode elements in window tensor into tf.float32
		return map(float, list(self.Alignment)) + map(lambda x:qual2code(x), list(self.Qual)) + map(float, list(self.Strand))

class Data_Reader():
	def __init__(self,handle,batch_size=BATCH_SIZE):
		self.handle = handle
		self.batch_size = batch_size
	def read_batch(self):
		res_window = [] # This list holds batch_size of window_tensor
		res_label = []
		i = 0
		#print "Readling One Batch"
		while i < self.batch_size:
			l = self.handle.readline()
			if l == '':
				self.handle.seek(0)
				continue
			else: 
				one_tensor = window_tensor(l.strip())
				res_window.append(one_tensor.encode())
				res_label.append(one_tensor.label)
				i += 1
		#print "Finish Reading"
		return res_window, res_label

# ARG: batch_size: The batch size will be baked into both placeholders.
# Return: Tensors placeholder, Labels placeholder.
def placeholder_inputs(batch_size):
	#tensor_placeholder = tf.placeholder(tf.float32, shape=(batch_size,WIDTH,HEIGHT+1,3))
	tensor_placeholder = tf.placeholder(tf.float32, shape=(batch_size,WIDTH*(HEIGHT+1)*3))
	labels_placeholder = tf.placeholder(tf.int32, shape = batch_size)
	return tensor_placeholder, labels_placeholder

def fill_feed_dict(data_set, tensor_pl, labels_pl):
	tensor_feed, labels_feed = data_set.read_batch()
	feed_dict = {
			tensor_pl: tensor_feed,
			labels_pl: labels_feed
			}
	return feed_dict


def GetOptions():
	parser = OptionParser()
	parser.add_option('-','--',dest = '', metavar = '', help = '')
	(options,args) = parser.parse_args()
	
	return

def main():
	#hand = gzip.open('/home/local/users/jw/TensorFlowCaller/Nebraska_NA12878_HG001_TruSeq_Exome/sample_1/windows_training.txt.gz','rb')
	hand = gzip.open('/home/local/users/jw/TensorFlowCaller/Nebraska_NA12878_HG001_TruSeq_Exome/sample_1/windows_testing.txt.gz','rb')
	Test_Data = Data_Reader(hand, 1000)
	start_time = time.time()
	Test_Data.read_batch()
	print time.time() - start_time
	return	

if __name__=='__main__':
	main()
