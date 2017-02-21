#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang explorerwjy@gmail.com

#========================================================================================================
# Calling Variants with saved model
#========================================================================================================


from datetime import datetime
import math
import time
import sys
import os
import numpy as np
import tensorflow as tf
import Window2Tensor
from Input import *
import Models

BATCH_SIZE = FLAGS.batch_size

def GetCheckPoint():
	ckptfile = FLAGS.checkpoint_dir+'/log/checkpoint'
	if not os.path.isfile(ckptfile):
		print "Model checkpoint not exists."
		exit()
	f = open(ckptfile,'rb')
	ckpt = f.readline().split(':')[1].strip().strip('"')
	f.close()
	prefix = os.path.abspath(FLAGS.checkpoint_dir+'/log/')
	ckpt = prefix + '/' + ckpt
	return ckpt

# dataset: Window2Tensor.Data_Reader object, read BATCH_SIZE samples a time.
def do_eval(sess, normed_logits, prediction, tensor_pl, label_pl, dataset, Total, fout):
	steps_per_epoch = Total // BATCH_SIZE
	num_examples = steps_per_epoch * BATCH_SIZE
	for step in xrange(steps_per_epoch):
		tensor, label = dataset.read_batch()
		feed_dict = {tensor_pl: tensor, label_pl: label}
		GL, GT = sess.run([normed_logits, prediction], feed_dict = feed_dict)
		for gt, gl in zip(GT, GL):
			#print gt, gl
			gl = map(str, gl)
			fout.write(str(gt)+'\t'+','.join(gl)+'\n')



def Calling(TrainingData, ValidationData, TestingData, ModelCKPT):
	Num_training = 3522409 
	Num_validation = 86504
	Num_testing = 186468
	#with tf.Graph().as_default() as g:
	with tf.device('/gpu:2'):
		TrainingData = gzip.open(TrainingData,'rb')
		ValidationData = gzip.open(ValidationData,'rb')
		TestingData = gzip.open(TestingData,'rb')

		fout_training = open('Calling_training.txt','wb')
		fout_validation = open('Calling_validation.txt','wb')
		fout_testing = open('Calling_testing.txt','wb')

		dataset_training = Window2Tensor.Data_Reader(TrainingData, batch_size=BATCH_SIZE)
		dataset_validation = Window2Tensor.Data_Reader(ValidationData, batch_size=BATCH_SIZE)
		dataset_testing = Window2Tensor.Data_Reader(TestingData, batch_size=BATCH_SIZE)
		TensorPL, LabelPL = Window2Tensor.placeholder_inputs(BATCH_SIZE)

		#TrainingTensor, TrainingLabel = dataset_training.read_batch()
		#ValidationTensor, ValidationLabel = dataset_validation.read_batch()
		#TestingTensor, TestingLabel = dataset_testing.read_batch()


		convnets = Models.ConvNets()
		logits = convnets.Inference(TensorPL)
		normed_logits = tf.nn.softmax(logits, dim=-1, name=None)
		prediction=tf.argmax(normed_logits,1)

		saver = tf.train.Saver() 

		config = tf.ConfigProto(allow_soft_placement = True)
		with tf.Session(config = config) as sess:
			saver.restore(sess, ModelCKPT)
			
			#print TrainingLabel
			#print sess.run(logits,feed_dict = {TensorPL:TrainingTensor})
			
			print "Evaluating On Training Sample"
			do_eval(sess, normed_logits, prediction, TensorPL, LabelPL, dataset_training, Num_training, fout_training)
			print "Evaluating On Vlidation Sample"
			do_eval(sess, normed_logits, prediction, TensorPL, LabelPL, dataset_validation, Num_validation, fout_validation)
			print "Evaluating On Testing Sample"
			do_eval(sess, normed_logits, prediction, TensorPL, LabelPL, dataset_testing, Num_testing, fout_testing)

		fout_training.close()
		fout_validation.close()
		fout_testing.close()

def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(FLAGS.eval_dir):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	
	# Get File Name of TraingData, ValidationData and Testdata
	TrainingData = FLAGS.TrainingData
	ValidationData = FLAGS.ValidationData
	TestingData = FLAGS.TestingData
	# Get The Saved Model
	# ModelCKPT = FLAGS.checkpoint_dir+'/model.ckpt-4599.meta'

	ModelCKPT = GetCheckPoint()
	Calling(TrainingData, ValidationData, TestingData, ModelCKPT)



if __name__ == '__main__':
	tf.app.run()
