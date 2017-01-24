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

BATCH_SIZE = 100

# ==========================================================================
# Encode Rule for window tensor
BASE = {'A':1, 'T':2, 'G':3, 'C':4, 'N':5, 'X':6, '.':0}
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
	def __init__(self,header):
		tmp = header.strip().split('\t')
		self.Alignment = []
		self.Qual = []
		self.Strand = []
		self.chrom,self.pos = tmp[1].split(':')
		self.label = tmp[4]
	def read_data(self,handle,which):
		for i in xrange(HEIGHT+1):
			l = handle.readline().strip()
			if which == 'Alignment':
				self.Alignment.append(l)		
			elif which == 'Qual':
				self.Qual.append(l)
			elif which == 'Strand':
				self.Strand.append(l)
			else:
				print "Error window Attribute. Check the Window2Tensor.py"
	def read_data_2(self,handle,which):
		for i in xrange(HEIGHT+1):
			l = list(handle.readline().strip())
			if which == 'Alignment':
				self.Alignment.append([base2code(base) for base in l ])		
				#self.Qual.append(l)
			elif which == 'Qual':
				self.Qual.append([qual2code(ch) for ch in l ])
				#self.Qual.append(l)
			elif which == 'Strand':
				self.Strand.append([strand2code(ch) for ch in l ])
				#self.Strand.append(l)
			else:
				print "Error window Attribute. Check the Window2Tensor.py"

	def encode(self):
		# This func encode elements in window tensor into tf.float32
		res = []
		# Encode bases
		Alignment = []
		Qual = []
		Strand = []
		for seq in self.Alignment:
			Alignment.extend(seq)
		#print 'Encode Read Finish'
		for seq in self.Qual:
			Qual.extend(seq)
		#print 'Encode qual Finish'
		for seq in self.Strand:
			Strand.extend(seq)
		#print 'Encode strand Finish'
		return Alignment + Qual + Strand

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
			elif l.startswith('#Alignment'): # start of a new window tensor
				one_tensor = window_tensor(l)
				#one_tensor.read_data(self.handle,'Alignment')
				one_tensor.read_data_2(self.handle,'Alignment')
				#print len(one_tensor.Alignment)
			elif l.startswith('#QUAL'):
				#one_tensor.read_data(self.handle,'Qual')
				one_tensor.read_data_2(self.handle,'Qual')
			elif l.startswith('#Strand'):
				#one_tensor.read_data(self.handle,'Strand')
				one_tensor.read_data_2(self.handle,'Strand')
				i += 1
				#print 'Read findish'

				# A window tensor loaded complete.
				res_window.append(one_tensor.encode())
				res_label.append(one_tensor.label)
				#print 'Encode finish'
			else:
				print l
				print "Window Shape Error, check input shape and HEIGHT"
				exit()
		#print "Finish Reading"
		return res_window, res_label
	def read_batch_2(self):
		res_window = []
		res_label = []
		i = 0
		while i < self.batch_size:
			self.handle.readline()
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
