#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Eval_Calling.py
# Eval on Tensor caller "called" variants.
# Need Wendow Tensor evaled from and called .txt file with GT
#========================================================================================================

from datetime import datetime
import gzip
import math
import time
import sys
import os
import numpy as np
import Region
#import tensorflow as tf
#import Window2Tensor
#from Input import *

class Counts():
	def __init__(self):
		self.zero_zero = 0
		self.zero_one = 0
		self.zero_two = 0
		self.one_zero = 0
		self.one_one = 0
		self.one_two = 0
		self.two_zero = 0
		self.two_one = 0
		self.two_two = 0
	def Get_POS_Eval(self):
		self.POS_TP = self.one_one + self.one_two + self.two_one + self.two_two
		self.POS_FP = self.one_zero + self.two_zero
		self.POS_FN = self.zero_one + self.zero_two
		self.POS_SE = float(self.POS_TP)/(self.POS_TP + self.POS_FN)
		self.POS_PPV = float(self.POS_TP)/(self.POS_TP+self.POS_FP)
		self.POS_F1 = float(2*self.POS_TP)/(2*self.POS_TP + self.POS_FN + self.POS_FP)
	def Get_Genotype_Eval(self):
		self.GT_TP = self.one_one + self.two_two
		self.GT_FP = self.one_two + self.two_one + self.one_zero + self.two_zero
		self.GT_FN = self.zero_one + self.zero_two
		self.GT_SE = float(self.GT_TP)/(self.GT_TP + self.GT_FN)
		self.GT_PPV = float(self.GT_TP)/(self.GT_TP+self.GT_FP)
		self.GT_F1 = float(2*self.GT_TP)/(2*self.GT_TP + self.GT_FN + self.GT_FP)
	def show(self):
		print '0/0 -> 0/0:',self.zero_zero
		print '0/0 -> 0/1:',self.zero_one
		print '0/0 -> 1/1:',self.zero_two
		print '0/1 -> 0/0:',self.one_zero
		print '0/1 -> 0/1:',self.one_one
		print '0/1 -> 1/1:',self.one_two
		print '1/1 -> 0/0:',self.two_zero
		print '1/1 -> 0/1:',self.two_one
		print '1/1 -> 1/1:',self.two_two
		print '-'*50
		print 'Position Eval:'
		print 'TP:',self.POS_TP
		print 'FP:',self.POS_FP
		print 'FN:',self.POS_FN
		print 'SE:',self.POS_SE
		print 'PPV:',self.POS_PPV
		print 'F1:',self.POS_F1
		print '-'*50
		print 'Genotype Eval:'
		print 'TP:',self.GT_TP
		print 'FP:',self.GT_FP
		print 'FN:',self.GT_FN
		print 'SE:',self.GT_SE
		print 'PPV:',self.GT_PPV
		print 'F1:',self.GT_F1

def Eval_Calling(TruthData,PredictedData):
	print "Eval on",PredictedData
	counts = Counts()
	fin1 = gzip.open(TruthData, 'rb')
	fin2 = open(PredictedData, 'rb')
	fout = open('More_Detailed_'+PredictedData,'wb')
	for l in fin2:
		P_label, GL = l.strip().split('\t')
		flag = fin1.readline()[:13]
		Label = flag[0]
		Chrom = Region.Byte2Chrom(flag[1:3])
		Pos = Region.Byte2Pos(flag[3:13])
		if P_label == '0' and Label == '0':
			counts.zero_zero += 1
		elif P_label == '0' and Label == '1':
			counts.zero_one += 1
		elif P_label == '0' and Label == '2':
			counts.zero_two += 1
		elif P_label == '1' and Label == '0':
			counts.one_zero += 1
		elif P_label == '1' and Label == '1':
			counts.one_one	+= 1
		elif P_label == '1' and Label == '2':
			counts.one_two += 1
		elif P_label == '2' and Label == '0':
			counts.two_zero += 1
		elif P_label == '2' and Label == '1':
			counts.two_one += 1
		elif P_label == '2' and Label == '2':
			counts.two_two += 1
	counts.Get_POS_Eval()
	counts.Get_Genotype_Eval()
	counts.show()


def main(argv=None):  # pylint: disable=unused-argument
	# Get File Name of TraingData, ValidationData and Testdata
	TrainingData = "Training.windows.txt.gz" #FLAGS.TrainingData
	ValidationData = "windows_validation.txt.gz" #FLAGS.ValidationData
	TestingData = "Testing.windows.txt.gz" #FLAGS.TestingData
	TrainingPreticted = 'Calling_training.txt'
	ValidationPreticted = 'Calling_validation.txt'
	TestingPreticted = 'Calling_testing.txt'

	Eval_Calling(TrainingData, TrainingPreticted)
	Eval_Calling(ValidationData, ValidationPreticted)
	Eval_Calling(TestingData, TestingPreticted)

if __name__=='__main__':
	main()

