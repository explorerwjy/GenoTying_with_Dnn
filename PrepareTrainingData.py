#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# PrepareTrainingData.py: 
# Use vcf file as input, Generate window for each candidate variant.
# Chr1 - Chr19 as training set, Chr20 - Chr22 as test set.
#
# If use Multi-processing, will produce splited data and then need to merge them back.
#========================================================================================================

import argparse
import pysam
from utils import *
import Region
import gzip
import multiprocessing
import time
import os
import subprocess
import sys

sys.stdout = sys.stderr

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r','--ref',type=str, default='"/home/local/users/jw/resources/references/b37/hg19.fasta"' ,help = 'Reference Genome')
	parser.add_argument('-b','--bam',type=str, help = 'Aligned bam file')
	parser.add_argument('-v','--vcf',type=str, help = 'Candidate variants to scan')
	parser.add_argument('-m','--mode',default=1, choices=[1,2], help = 'Mode. 1.VarScan 2.OneVar')
	parser.add_argument('-t','--true',type=str, help = 'True Positive Variant VCF')
	parser.add_argument('-p','--process', type=int, default=1, help='Num of process used to produce result')

	args = parser.parse_args()
	
	return args.ref,args.bam,args.vcf,args.true,args.mode,args.process 

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
			k,p,v = var2kv2(l)
			if k not in res:
				res[k] = v
			else:
				print "Multiple record in %s has same position: %s"%(vcf,p)
	return res

# Scan a candidate vcf file, generate window for the variant and mark genotype according to GIAB positives
def VarScan(referenceGenome,bam,Candidate_vcf,Positive_vars,Nprocess):
	jobs = []
	for i in range(Nprocess):
		p = multiprocessing.Process(target=load_variants, args=(Candidate_vcf, Positive_vars, referenceGenome, bam, i, Nprocess))
		jobs.append(p)
		p.start()
	for job in jobs:
		job.join()
	# Merge all files
	print "Merging Files together and bgzip"
	command1 = 'cat tmp.train.*.windows.txt| sort -k1,1d -k2,2n > Training.windows.txt ;bgzip -f Training.windows.txt; tabix -f -s 1 -b 2 -e 3 Training.windows.txt.gz'
	command2 = 'cat tmp.test.*.windows.txt| sort -k1,1d -k2,2n > Testing.windows.txt ;bgzip -f Testing.windows.txt; tabix -f -s 1 -b 2 -e 3 Testing.windows.txt.gz'
	process1 = subprocess.Popen(command1, shell=True, stdout=subprocess.PIPE)
	process2 = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE)
	process1.wait()
	process2.wait()
	print process1.returncode
	print process2.returncode
	process3 = subprocess.Popen('rm tmp.*.windows.txt', shell=True, stdout=subprocess.PIPE)
	process3.wait()
	print process3.returncode

def load_variants(VCF, Positive_vars, referenceGenome, bam, i, n):
	outname_train = 'tmp.train.'+str(i)+'.windows.txt'
	outname_test = 'tmp.test.'+str(i)+'.windows.txt'
	fout_train = open(outname_train, 'wb')
	fout_test = open(outname_test,'wb')
	window_generator = parse_tabix_file_subset([VCF], Positive_vars, referenceGenome, bam, i, n, get_variants_from_sites_vcf)
	for record in window_generator:
		if record.chrom not in ['20','21','22','X','Y']:
			fout_train.write(record.write())
		elif record.chrom in ['20','21','22']:
			fout_test.write(record.write())

def parse_tabix_file_subset(tabix_filenames, Positive_vars, referenceGenome, bam, subset_i, subset_n, record_parser):
	start_time = time.time()
	open_tabix_files = [pysam.Tabixfile(tabix_filename) for tabix_filename in tabix_filenames]
	tabix_file_contig_pairs = [(tabix_file, contig) for tabix_file in open_tabix_files for contig in tabix_file.contigs]
	tabix_file_contig_subset = tabix_file_contig_pairs[subset_i : : subset_n]
	short_filenames = ",".join(map(os.path.basename, tabix_filenames))
	num_file_contig_pairs = len(tabix_file_contig_subset)
	print "Lodaing subset %d from %d" % (subset_i, subset_n)
	
	RefFile = pysam.FastaFile(referenceGenome)
	SamFile = samfile = pysam.AlignmentFile(bam, "rb")
	
	counter = 0
	for tabix_file, contig, in tabix_file_contig_subset:
		#header_iterator = tabix_file.header
		records_iterator = tabix_file.fetch(contig, 0, 10**9, multiple_iterators=True)
		#for parsed_record in record_parser(itertools.chain(header_iterator, records_iterator), Positive_vars, RefFile, SamFile ):
		for parsed_record in record_parser(records_iterator, Positive_vars, RefFile, SamFile ):
			counter += 1
			yield parsed_record

			if counter % 10000 == 0:
				seconds_elapsed = float(time.time()-start_time)
				print "Load %d records from subset %d of %d from %s in %f seconds" % (counter, subset_i, subset_n, short_filenames, seconds_elapsed)

# The record_parser in parse_tabix_file_subset
def get_variants_from_sites_vcf(sites_file, Positive_vars, RefFile, SamFile):
	for l in sites_file:
		#if l.startswith('##'):
		#	continue
		#elif l.startswith('#'):
		#	continue
		llist = l.strip().split('\t')
		k,chrom,pos,ref,alt = var2kv(llist)
		if k in Positive_vars:
			GT = get_Genotype(llist)
			region = Region.CreateRegion(RefFile, SamFile, chrom, pos, ref, alt, str(GT)) #Create a Region according to a site
		else:
			region = Region.CreateRegion(RefFile, SamFile, chrom, pos, ref, alt, '0') 
		yield region


# This func used to view window in a given region. Mainly aimed to debug the Region part.
def OneVar(ref,bam): 
	RefFile = pysam.FastaFile(ref)
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
	referenceGenome,bam,vcf,T_vcf,mode,Nprocess = GetOptions()
	if mode == '2':
		OneVar(bam)
	else:
		if T_vcf == None:
			print "Please provide Positive Data"
		Positives = Get_Positives(T_vcf)
		VarScan(referenceGenome,bam,vcf,Positives,Nprocess)
	

if __name__=='__main__':
	main()
