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

def var2kv2(l):
	llist = l.strip().split('\t')
	chrom,pos = llist[0:2]
	p = chrom+':'+pos
	k = get_xpos(chrom,pos)
	return k,p,l

def var2kv(llist):
	chrom,pos = llist[0:2]
	ref,alt = llist[3:5]
	#p = chrom+'-'+pos
	k = get_xpos(chrom,pos)
	return k,chrom,pos,ref,alt

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

def Chrom2Byte(chrom):
	chrom = str(chrom)
	encode = {'1':'01', '2':'02', '3':'03', '4':'04', '5':'05', '6':'06', '7':'07', '8':'08', '9':'09', '10':'10', '11':'11', '12':'12', '13':'13', '14':'14', '15':'15', '16':'16', '17':'17', '18':'18', '19':'19', '20':'20', '21':'21', '22':'22', 'X':'23', 'Y':'24'}
	return encode[chrom]
def Byte2Chrom(Byte):
	decode = {'01':'1', '02':'2', '03':'3', '04':'4', '05':'5', '06':'6', '07':'7', '08':'8', '09':'9','10':'10', '11':'11', '12':'12', '13':'13', '14':'14', '15':'15', '16':'16', '17':'17', '18':'18', '19':'19', '20':'20', '21':'21', '22':'22', '23':'X', '24':'Y'}
	return decode[Byte]

def Pos2Byte(pos):
	pos = str(pos)
	if len(pos) < 10:
		return '0'*(10-len(pos))+pos
	elif len(pos) > 10:
		print pos
		exit()
	else:
		return pos
def Byte2Pos(pos):
	return str(int(pos))

def GetOptions():
	parser = OptionParser()
	parser.add_option('-','--',dest = '', metavar = '', help = '')
	(options,args) = parser.parse_args()
	
	return

def main():
	pass	

if __name__=='__main__':
	main()
