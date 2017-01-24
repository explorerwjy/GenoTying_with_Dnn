#!/home/local/users/jw/bin/python2.7
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# PrepareTrainingData.py: 
# Use vcf file as input, Generate window for each candidate variant.
# Chr1 - Chr19 as training set, Chr20 - Chr22 as test set.
#========================================================================================================

from optparse import OptionParser
import pysam
from utils import *
#import Variant
import Region
import gzip

def GetOptions():
	parser = OptionParser()
	parser.add_option('-b','--bam',dest = 'BAM', metavar = 'BAM', help = 'Aligned bam file')
	parser.add_option('-v','--var',dest= 'VCF', metavar = 'VCF', help = 'Candidate variants to scan')
	parser.add_option('-m','--mode',dest= 'mode', metavar = 'mode', help = 'Mode. 1.VarScan 2.OneVar')
	parser.add_option('-t','--true',dest= 'true', metavar = 'true', help = 'True Positive Variant VCF')
	(options,args) = parser.parse_args()
	
	return options.BAM,options.VCF,options.true,options.mode

def Get_Positives(T_vcf):
	if T_vcf.endswith('.vcf.gz'):
		fin = gzip.open(T_vcf)
	else:
		fin = open(T_vcf)
	res = {}
	for l in fin:
		if l.startswith('#'):
			continue
		else:
			k,p,v = var2kv(l)
			if k not in res:
				res[k] = v
			else:
				print "Multiple record in %s has same position: %s"%(vcf,p)
	return res

# Scan a candidate vcf file, generate window for the variant and mark genotype according to GIAB positives
def VarScan(bam,Candidate_vcf,Positive_vars):
	RefFile = pysam.FastaFile("/home/local/users/jw/resources/references/b37/hg19.fasta")
	SamFile = samfile = pysam.AlignmentFile(bam, "rb")
	fout_training = gzip.open('windows_training.txt.gz','wb')
	fout_testing = gzip.open('windows_testing.txt.gz','wb')
	fin = open(Candidate_vcf,'rb')
	for l in fin:
		if l.startswith('##'):
			continue
		elif l.startswith('#'):
			header = l.strip().split('\t')
		else:	
			llist = l.strip().split('\t')
			chrom, pos = llist[0:2]
			if chrom not in ['20','21','22','X','Y']:	
				k,p,v = var2kv(l)
				#region = Region.Region(ref,samfile, chrom, int(pos))
				if k in Positive_vars:
					GT = get_Genotype(llist)
					region = Region.CreateRegion(RefFile, SamFile, chrom, pos, str(GT)) #Create a Region according to a site
				else:
					region = Region.CreateRegion(RefFile, SamFile, chrom, pos, '0') 
				#Pulse(region)
				fout_training.write(region.write()+'\n')
			elif chrom in ['20','21','22']:
				k,p,v = var2kv(l)
				#region = Region.Region(ref,samfile, chrom, int(pos))
				if k in Positive_vars:
					GT = get_Genotype(llist)
					region = Region.CreateRegion(RefFile, SamFile, chrom, pos, str(GT)) #Create a Region according to a site
				else:
					region = Region.CreateRegion(RefFile, SamFile, chrom, pos, '0') 
				#Pulse(region)
				fout_testing.write(region.write()+'\n')
	fout_training.close()
	fout_testing.close()

# This func used to view window in a given region. Mainly aimed to debug the Region part.
def OneVar(bam): 
	RefFile = pysam.FastaFile("/home/local/users/jw/resources/references/b37/hg19.fasta")
	SamFile = samfile = pysam.AlignmentFile(bam, "rb")
	while 1:
		tmp = raw_input('Please enter the chr:pos >> ').split(':')
		if len(tmp) == 2:	
			chrom,pos = tmp
			region = Region.CreateRegion(RefFile, SamFile, chrom, pos, 'N', True) #Create a Region according to a site
			region.show()
		elif tmp[0] == 'N' or tmp[0] == 'n':
			exit()
		else:
			print tmp
			print tmp.split(':')
			exit()
def Pulse(region):
	region.show()
	ans = raw_input("Go to next var (y/n)? >>")
	if ans.lower() == 'y':
		return
	else:
		exit()

def main():
	bam,vcf,T_vcf,mode = GetOptions()
	if mode == '2':
		OneVar(bam)
	else:
		if T_vcf == None:
			T_vcf = '/home/local/users/jw/TensorFlowCaller/Nebraska_NA12878_HG001_TruSeq_Exome/Positive.norm.vcf'
		Positives = Get_Positives(T_vcf)
		VarScan(bam,vcf,Positives)
	

if __name__=='__main__':
	main()
