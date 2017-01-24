#!/home/local/users/jw/bin/python2.7
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
#
#========================================================================================================

from optparse import OptionParser
import re

CHROMOSOMES = ['chr%s' % x for x in range(1,23)]
CHROMOSOMES.extend(['chrX', 'chrY', 'chrM'])
CHROMSOME2CODE = {item: i+1 for i ,item in enumerate(CHROMOSOMES)}

def get_xpos(chrom, pos):
	if not chrom.startswith('chr'):
		chrom = 'chr{}'.format(chrom)
	return CHROMSOME2CODE[chrom] * int(1e9) + int(pos)

def var2kv(l):
	llist = l.split('\t')
	chrom,pos = llist[0:2]
	p = chrom+'-'+pos
	k = get_xpos(chrom,pos)
	return k,p,l

def get_Genotype(llist):
	#fmt = llist[9]
	#data = llist[10]
	GT = re.findall('[\d.]',llist[9].split(':')[0])
	if GT[0] == '0' and GT[1] == '0':
		return 0 # Homozygous Ref
	elif GT[0] != GT[1]:
		return 1 # Hetrozygous
	else:
		return 2 # Homozygous Alt

def GetOptions():
	parser = OptionParser()
	parser.add_option('-','--',dest = '', metavar = '', help = '')
	(options,args) = parser.parse_args()
	
	return

def main():
	pass	

if __name__=='__main__':
	main()
