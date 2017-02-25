#!/home/local/users/jw/bin/python2.7
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Make a evaluation of candidate variants and positive variants
# Training: chr1-19
# Testing: chr20-22
#========================================================================================================

import argparse
import union_candidates as uc
import gzip

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--positive', help = 'VCF file contains Positive variants')
	parser.add_argument('-c','--candidate', help = 'VCF file contains Candidate variants')
	parser.add_argument('-m','--mode',type=str, choices=['train','test','all'], help='mode for run. train will load chr1-19 from positive vcf. test will load chr20-22 from positive vcf. all will load all variants from positive vcf')
	args = parser.parse_args()
	
	return args.Positive,args.Candidate
def getTPFP(candidates,positives):
	TP,FP = 0,0

	fout_tp = open('tp.vcf','wb')
	fout_fp = open('fp.vcf','wb')
	
	for k,v in candidates.items():
		if k in positives:
			TP += 1
			fout_tp.write(v)
		else:
			FP += 1
			fout_fp.write(v)
	fout_tp.close()
	fout_fp.close()
	return TP,FP
def getFN(candidates,positives):
	FN = 0
	fout_fn = open('fn.vcf','wb')
	for k,v in positives.items():
		if k not in candidates:
			FN += 1
			fout_fn.write(v)
	fout_fn.close()
	return FN
def Evaluation(PositiveVCF,CandidateVCF):
	#fin_P = open(PositiveVCF,'rb')
	#fin_C = open(CandidateVCF,'rb')
	meta,header,candidates = uc.Parse_one(CandidateVCF)
	meta,header,positives = uc.Parse_one(PositiveVCF)
	TP,FP = getTPFP(candidates,positives)
	FN = getFN(candidates,positives)
	
	SE = float(TP)/(TP+FN)
	PPV = float(TP)/(TP+FP)
	F1 = float(2*TP)/(2*TP + FN + FP)
	print "N candidates: %d\tN positives: %d"%(len(candidates.keys()),len(positives.keys()))
	print "TP: %d\tFP: %d\tFN: %d"%(TP,FP,FN)
	print "SE: %f\tPPV: %f\tF1: %f"%(SE,PPV,F1)

def main():
	PositiveVCF,CandidateVCF = GetOptions()
	Evaluation(PositiveVCF,CandidateVCF)
	

if __name__=='__main__':
	main()
