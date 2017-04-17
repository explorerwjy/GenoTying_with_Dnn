#!/home/local/users/jw/bin/python2.7
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# TensorCaller: Calling Variants from Tensor represented Aligned Reads
#=========================================================================

from optparse import OptionParser
import pysam
#import utils
#import Variant
import Region


def GetOptions():
    parser = OptionParser()
    parser.add_option(
        '-b',
        '--bam',
        dest='BAM',
        metavar='BAM',
        help='Aligned bam file')
    parser.add_option(
        '-v',
        '--var',
        dest='VCF',
        metavar='VCF',
        help='Candidate variants to scan')
    (options, args) = parser.parse_args()

    return options.BAM, options.VCF


def VarScan(bam, vcf):
    RefFile = pysam.FastaFile(
        "/home/local/users/jw/resources/references/b37/hg19.fasta")
    SamFile = samfile = pysam.AlignmentFile(bam, "rb")
    fin = open(vcf, 'rb')
    for l in fin:
        if l.startswith('##'):
            continue
        elif l.startswith('#'):
            header = l.strip().split('\t')
        else:
            llist = l.strip().split('\t')
            chrom, pos = llist[0:2]
            #region = Region.Region(ref,samfile, chrom, int(pos))
            # Create a Region according to a site
            region = Region.CreateRegion(RefFile, SamFile, chrom, pos)
            region.show()

            Pulse()


def Pulse():
    ans = raw_input("Go to next var (y/n)? >>")
    if ans.lower() == 'y':
        return
    else:
        exit()


def main():
    bam, vcf = GetOptions()
    VarScan(bam, vcf)


if __name__ == '__main__':
    main()
