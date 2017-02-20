#!/home/local/users/jw/bin/python2.7
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

from optparse import OptionParser
import gzip
import re

def GetOptions():
	parser = OptionParser()
	parser.add_option('-t','--truth',dest = 'Truth', metavar = 'Truth', help = 'VCF file contains All Positive variants')
	parser.add_option('-c','--candidate',dest = 'Candidate', metavar = 'Candidate', help = 'VCF file contains Candidate variants')
	(options,args) = parser.parse_args()
	return options.Truth,options.Candidate

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
		self.GT_PPV = float(self.GT_TP)/(self.GT_TP+self.GTS_FP)
		self.GT_F1 = float(2*self.GT_TP)/(2*self.GT_TP + self.GT_FN + self.GT_FP)

def var2kv(l):
	llist = l.strip().split('\t')
	chrom, pos = llist[0:2]
	alleles = llist[3:5]
	gt = llist[9].split(':')[0]
	gt = re.findall('[\d.]',gt)
	gt.sort()
	k = chrom + '-' + pos
	v = [ alleles[gt[0]], alleles[gt[1]] ]
	return k, v

def Classify(l, True_dict, counts):
	k, v = var2kv(l)
	if k in True_dict:
		if v[0] == True_dict[k][0] and v[1] == True_dict[k][1] and v[0] == v[1]: #Homozygous Alt
			counts.two_two += 1
		elif v[0] == True_dict[k][0] and v[1] == True_dict[k][1] and v[0] != v[1]: #Hetrozygous
			counts.one_one += 1
		elif v[1] == True_dict[k][1] and v[0] != True_dict[k][0] and True_dict[k][0] == True_dict[k][1]: # 0/1 - 1/1
			counts.one_two += 1
		elif v[0] == v[1] and v[1] == True_dict[k][1] and True_dict[k][0] != True_dict[k][1]: # 1/1 - 0/1
			counts.two_one += 1
		True_dict.pop(k)
	else:
		if v[0] == v[1]:
			counts.two_zero += 1
		elif v[0] != v[1]:
			counts.one_zero += 1

def GetFNs(counts, True_dict):
	for k,v in True_dict.items():
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
				print "Multiple record in %s has same position: %s"%(vcf,p)
	return res

def Evaluation(PositiveVCF,CandidateVCF):
	True_dict = GetTruthDict(PositiveVCF)
	fin = GetHand(CandidateVCF)
	counts = Counts()
	for l in fin:
		if l.startswith('##'):
			continue
		elif l.startswith("#"):
			header = l
		else:
			Classify(l, True_dict, counts)
	GetFNs(counts,True_dict)
	counts.Get_POS_Eval()
	counts.Get_Genotype_Eval()
	counts.Show()

def main():
	PositiveVCF,CandidateVCF = GetOptions()
	Evaluation(PositiveVCF,CandidateVCF)
	

if __name__=='__main__':
	main()
