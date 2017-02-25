#!/home/local/users/jw/bin/python2.7
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Take Union of several candidate vcf file into one to increase sensitivity.
# UnFinished One
#========================================================================================================

import argparse
import gzip

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v','--vcf', nargs='+', type=str, help = 'vcf files to be union together')
	args = parser.parse_args()
	vcflist = [ vcf.strip() for vcf in args ]
	outname = 'Union_'+'-'.join([ vcf.split('/')[-1].split('.')[0].strip() for vcf in vcflist])+'.vcf'
	return vcflist, outname

# Return a uniq pos according to chrom and pos that easy to sort
CHROMOSOMES = ['chr%s' % x for x in range(1,23)]
CHROMOSOMES.extend(['chrX', 'chrY', 'chrM'])
CHROMSOME2CODE = {item: i+1 for i ,item in enumerate(CHROMOSOMES)}
def get_xpos(chrom, pos):
	if not chrom.startswith('chr'):
		chrom = 'chr{}'.format(chrom)
	return CHROMSOME2CODE[chrom] * int(1e9) + int(pos)

# turn a vcf record into a k,v pair
def var2kv(l):
	llist = l.split('\t')
	chrom,pos = llist[0:2]
	#p = chrom+'-'+pos
	k = get_xpos(chrom,pos)
	return k,l

# Parse one vcf file into meta, header, and a dictionary of variants.
def VCF2DICT(vcf):
	meta = []
	header = None
	variants = {}
	fin = open(vcf,'rb')
	for l in fin:
		if l.startswith('##'):
			meta.append(l)
		elif l.startswith("#"):
			header = l
		else:
			k,l = var2kv(l)
			if k not in variants:
				variants[k] = v
			else:
				#raise KeyError("Multiple record in %s has same position: %s"%(vcf,p))
				print "Multiple record in %s has same position: %s"%(vcf,p)
	return meta,header,variants

# Take Union of multiple vcf file
def Union(vcflist,outname):
	TMP = []
	hands = []
	fout = open(outname,'wb')
	for vcf in vcflist:
		if vcf.endswith('.vcf'):
			hands.append(open(vcf,'rb'))	
		elif vcf.endswith('.vcf.gz'):
			hands.append(gzip.open(vcf,'rb'))
		else:
			raise IOError('Only Accept vcf or vcf.gz files')

	# Init the buffers to read each vcf
	buffers = [] 
	pointers = []
	meta = []
	for hand in hands:
		for l in hand:
			if l.startswith('##'):
				meta.append(l)
			elif l.startswith("#"):
				header = l
			else:
				buffers.append(l)
				pointers.append(var2kv(l))
				break

	while HasBuffer(buffers):
			



	fout.close()
def HasBuffer(buffers):
	flag = False
	for item in buffers:
		flag = flag or (item != "")
	return flag
def main():
	vcflist,outname = GetOptions()
	Union(vcflist,outname)

if __name__=='__main__':
	main()
