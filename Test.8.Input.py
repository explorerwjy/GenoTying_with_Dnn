#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Training The ConvNet for Tensor Caller
#========================================================================================================

import argparse
from datetime import datetime
import time
import os 
from threading import Thread
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
#from Queue import Queue
import numpy as np
import tensorflow as tf
import Models
from Input import *
import sys
import pysam
sys.stdout = sys.stderr

BATCH_SIZE=FLAGS.batch_size
log_dir = FLAGS.log_dir
max_steps = FLAGS.max_steps
#DataDecodingQueue = Queue(4*1000*BATCH_SIZE)

class InputDataProducer():
	def __init__(self, FileName, batch_size, nprocs, DataDecodingQueue):
		self.hand = gzip.open(FileName,'rb')
		self.min_Size = batch_size*4 
		self.batch_size = batch_size
		self.nprocs = nprocs
		self.DataDecodingQueue = DataDecodingQueue
	def load(self):
		#global DataDecodingQueue
		res = []
		if self.DataDecodingQueue.qsize() < self.min_Size: 
			print 'Starting Data Reader. Current Queue length: %d' % self.DataDecodingQueue.qsize()
			len_batch_size = 10*self.batch_size
			procs = []
			s_time = time.time()
			for i in xrange(self.nprocs):
				lineBatch = self.readBatch(len_batch_size)
				p = multiprocessing.Process(target=self.worker, args=(lineBatch, self.DataDecodingQueue,))
				p.daemon = True
				procs.append(p)
				p.start()
			for i in xrange(self.nprocs):
				res.append(self.DataDecodingQueue.get())
			for p in procs:
				print p
				p.join()
				print p
			print '2'
			print "Done Decoding a Batch with %d samples, Costs %.3f secs. Current Queue length: %d" % (self.nprocs*len_batch_size, time.time()-s_time, self.DataDecodingQueue.qsize())
		else:
			return
	def readBatch(self, Batchsize):
		res = []
		for i in xrange(Batchsize):
			line = self.hand.readline()
			if line == '':
				self.hand.seek(0)
				line = self.hand.readline()
			res.append(line)
		return res
	def worker(self, lineBatch, DataDecodingQueue):
		for line in lineBatch:
			record = window_tensor(line)
			record.encode()
			DataDecodingQueue.put((record.res, record.label))
		print DataDecodingQueue.qsize()
		return 1 
	def get(self):
		global DataDecodingQueue
		data = []
		label = []
		for i in xrange(self.batch_size):
			one_data, one_label = DataDecodingQueue.get()
			data.append(one_data)
			label.append(onelabel)
		return data, label
def train():
	# Try multiProcess
	dtype = tf.float16 if FLAGS.use_fl16 else tf.float32
	BATCH_SIZE = FLAGS.batch_size
	
	print "Start Building Graph"
	with tf.Graph().as_default():
		print "Define Training Data FIFOQueue"
		queue_input_data = tf.placeholder(dtype, shape=[BATCH_SIZE, DEPTH * (HEIGHT+1) * WIDTH])
		queue_input_label = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		
		FileName = 'TestInput.txt.gz'
		DataDecodingQueue = Queue()
		DataProducer = InputDataProducer(FileName, BATCH_SIZE, 4, DataDecodingQueue)	
		min_loss = 100
		try:	
			for step in xrange(max_steps):
				DataProducer.load()
				start_time = time.time()
				if DataDecodingQueue.qsize() > BATCH_SIZE:
					data_batch, label_batch = DataProducer.get()
					curr_data, curr_labels, NumQ = sess.run([input_data_pl, input_label_pl], feed_dict={input_data_pl:data_batch, input_label_pl:label_batch})
					print "1 Batch Time Costs: %.3f, 1 Batch size: %d" % ((time.time() - start_time), len(curr_labels))

		#except Exception, e:
			#print e
		except KeyError:	
			exit()

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--Continue", help="continue training from a checkpoint",
                    type=str)
	args = parser.parse_args()
	if	args.Continue != None:

		if args.Continue.lower() in ['y', 'yes', 't', 'true']:
			return True
		else:
			return False
	else:
		return False

def main(argv=None):  # pylint: disable=unused-argument
	Continue = GetOptions()

	print 'TraingDir is:',FLAGS.train_dir
	if Continue == True:
		ckptfile = FLAGS.checkpoint_dir+'/log/checkpoint'
		f = open(ckptfile,'rb')
		ckpt = f.readline().split(':')[1].strip().strip('"')
		f.close()
		prefix = os.path.abspath(FLAGS.checkpoint_dir+'/log/')
		ckpt = prefix + '/' + ckpt
		print ckpt
		continue_train(ckpt)
	else:
		train()

if __name__ == '__main__':
	tf.app.run()
