#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Make a Detailed evaluation of called variants and positive variants
# Called\Truth  	0/0		0/1		1/1		Total
#			0/0  
#			0/1
#			1/1
#		  Total
#
# Store the Truth variant in a dictionary {Pos: allele1-allele2}
# A candidate variant can be: 
# 1. False Positive : Pos not in Truth
# 2. Genotype Error : Pos in Truth but allele1-allele2 not consistent with Truth
# 3. Genotype True  : Pos in Truth and allele1-allele2 also consistent with Truth
#========================================================================================================

import argparse
import gzip
import re

def GetOptions():
	(options,args) = parser.parse_args()
	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--positive', help = 'VCF file contains Positive variants')
	parser.add_argument('-c','--candidate', help = 'VCF file contains Candidate variants')
	parser.add_argument('-m','--mode',type=str, choices=['train','test','all'], help='mode for run. train will load chr1-19 from positive vcf. test will load chr20-22 from positive vcf. all will load all variants from positive vcf')
	args = parser.parse_args()
	return args.positive,options.candidate,options.mode

def GetHand(filename):
	if filename.endswith('.vcf.gz'):
		return gzip.open(filename,'rb')
	elif filename.endswith('.vcf'):
		return open(filename,'rb')
	else:
		print "Error with file name, must be .vcf or .vcf.gz"
		exit()

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
		print 'Eval Results on TestSet -> GroundTruth'
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

def var2kv(l):
	llist = l.strip().split('\t')
	chrom, pos = llist[0:2]
	ref, alt = llist[3:5]
	ref = [ref]
	alts = alt.split(',')
	alleles = ref + alts
	gt = llist[9].split(':')[0]
	gt = map(int,re.findall('[\d.]',gt))
	gt.sort()
	k = chrom + '-' + pos
	v = [ alleles[gt[0]], alleles[gt[1]] ]
	return k, v

def Classify(l, True_dict, counts, TP_hand, FP_hand, FN_hand):
	k, v = var2kv(l)
	if k in True_dict:
		if v[0] == True_dict[k][0] and v[1] == True_dict[k][1] and v[0] == v[1]: #Homozygous Alt
			counts.two_two += 1
			TP_hand.write(l)
		elif v[0] == True_dict[k][0] and v[1] == True_dict[k][1] and v[0] != v[1]: #Hetrozygous
			counts.one_one += 1
			TP_hand.write(l)
		elif v[1] == True_dict[k][1] and v[0] != True_dict[k][0] and True_dict[k][0] == True_dict[k][1]: # 0/1 - 1/1
			counts.one_two += 1
			FP_hand.write(l)
		elif v[0] == v[1] and v[1] == True_dict[k][1] and True_dict[k][0] != True_dict[k][1]: # 1/1 - 0/1
			counts.two_one += 1
			FP_hand.write(l)
		True_dict.pop(k)
	else:
		if v[0] == v[1]:
			counts.two_zero += 1
			FP_hand.write(l)
		elif v[0] != v[1]:
			counts.one_zero += 1
			FP_hand.write(l)

def GetFNs(counts, True_dict, FN_hand):
	for k,v in True_dict.items():
		FN_hand.write(k+'\t'+'\t'.join(v)+'\n')
		if v[0] == v[1]:
			counts.zero_two += 1
		elif v[0] != v[1]:
			counts.zero_one += 1


def GetTruthDict(PositiveVCF):
	res = {}
	fin = GetHand(PositiveVCF)
	for l in fin:
		if l.startswith('##'):
			continue
		elif l.startswith("#"):
			header = l
		else:
			k, v = var2kv(l)
			if k not in res:
				res[k] = v
			else:
				#raise KeyError("Multiple record in %s has same position: %s"%(vcf,p))
				print "Multiple record in %s has same position: %s"%(PositiveVCF.split('/')[-1],k)
	return res

def Evaluation(PositiveVCF,CandidateVCF):
	True_dict = GetTruthDict(PositiveVCF)
	fin = GetHand(CandidateVCF)
	basename = CandidateVCF.split('/')[-1].rstrip('.gz').rstrip('.vcf')
	TP_hand = open(basename+'_TP.vcf','wb')
	FP_hand = open(basename+'_FP.vcf','wb')
	FN_hand = open(basename+'_FN.vcf','wb')
	counts = Counts()
	for l in fin:
		if l.startswith('##'):
			continue
		elif l.startswith("#"):
			header = l
		else:
			try:
				Classify(l, True_dict, counts, TP_hand, FP_hand, FN_hand)
			except:
				print l
	GetFNs(counts,True_dict, FN_hand)
	counts.Get_POS_Eval()
	counts.Get_Genotype_Eval()
	counts.show()

def main():
	PositiveVCF,CandidateVCF = GetOptions()
	Evaluation(PositiveVCF,CandidateVCF)
	

if __name__=='__main__':
	main()
