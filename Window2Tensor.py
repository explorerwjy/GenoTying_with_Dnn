#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Read window from file and transform them into tensor
#========================================================================================================

from optparse import OptionParser
import tensorflow as tf
import numpy as np
import gzip
import time
import math
from Region import *
from Input import *


BATCH_SIZE = FLAGS.batch_size

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
		self.label = line[0]
		self.Alignment = line[ 13 : 13 + WIDTH * (HEIGHT+1) ]
		self.Qual = line[ 13 + WIDTH * (HEIGHT+1) : 13 + WIDTH * (HEIGHT+1)*2]
		self.Strand = line[13 + WIDTH * (HEIGHT+1)*2 : 13 + WIDTH * (HEIGHT+1)*3]

	def encode(self):
		# This func encode elements in window tensor into tf.float32
		
		#print len(self.Alignment), len(self.Alignment[0])
		#print len(self.Qual), len(self.Qual[0])
		#print len(self.Strand), len(self.Strand[0])
		#exit()
		res = [ (float(base)/6 - 0.5) for base in list(self.Alignment)] + 
			  [ qual2code(x) for x in list(self.Qual)] + 
			  [ float(x)/2-0.5 for x in list(self.Strand)] 
		#print len([float(x)/2-0.5 for x in list(self.Strand)]), len(list(self.Strand))
		return res

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
