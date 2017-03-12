#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Test.py
#========================================================================================================

import argparse
import gzip
import time
import decodeline

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument('-','--', type=str, help = '')
	args = parser.parse_args()
	
	return

def decode(line, WIDTH, HEIGHT):
	chrom, start, end, ref, alt, label, window = line.strip().split('\t')
	Alignment = window[ 0 : WIDTH * (HEIGHT+1) ]
	Qual = window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
	Strand = window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
	p1 = [float(x) for x in Alignment]
	p2 = [((float(ord(x) - 33) / 60) - 0.5) for x in Qual]
	p3 = [float(x) for x in Strand]
	return p1 + p2 + p3

def Test_python():
	fname = '/home/yufengshen/TensorFlowCaller/data/NA12878/WGS_S2/TestInput.txt.gz'
	fin = gzip.open(fname)
	for l in fin:
		

def main():
	Test_python()
	Test_Cython()
	return

if __name__=='__main__':
	main()
