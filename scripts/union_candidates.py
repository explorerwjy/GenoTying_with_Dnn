#!/home/local/users/jw/bin/python2.7
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# Take Union of several candidate vcf file into one to increase sensitivity.
#========================================================================================================

from optparse import OptionParser



def GetOptions():
	parser = OptionParser()
	parser.add_option('-v','--vcf',dest = 'VCFs', metavar = 'VCFs', help = 'vcf files to union together')
	(options,args) = parser.parse_args()
	vcflist = [ vcf.strip() for vcf in options.VCFs.split(',') ]
	outname = 'union_'+'-'.join([ vcf.split('/')[-1].strip('.vcf') for vcf in vcflist])+'.vcf'
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
	p = chrom+'-'+pos
	k = get_xpos(chrom,pos)
	return k,p,l
# Parse one vcf file into meta, header, and a dictionary of variants.
def Parse_one(vcf):
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
			k,p,v = var2kv(l)
			if k not in variants:
				variants[k] = v
			else:
				#raise KeyError("Multiple record in %s has same position: %s"%(vcf,p))
				print "Multiple record in %s has same position: %s"%(vcf,p)
	return meta,header,variants
# Take Union of multiple vcf file
def Union(vcflist,outname):
	TMP = []
	fout = open(outname,'wb')
	for vcf in vcflist:
		meta,header,variants = Parse_one(vcf)
		fout.write(''.join(meta))
		TMP.append(variants)
	Union = {}
	for batch in TMP:
		for k,v in batch.items():
			if k not in Union:
				Union[k] = v
			else:
				# Make a compare of them
				continue
	#sort by xpos
	res = sorted(Union.items())
	fout.write(header)
	for k,v in res:
		fout.write(v)
	fout.close()

def main():
	vcflist,outname = GetOptions()
	Union(vcflist,outname)

if __name__=='__main__':
	main()
