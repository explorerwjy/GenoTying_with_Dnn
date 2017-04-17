#!/home/yufengshen/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# LocalAssembleRegion.py
# Load the GATK callset and Other callset.
# If Indel present, select the pos and padding 500 bp, output the Region.
#=========================================================================

import argparse
import sys
import gzip
import time

PADDING_SIZE = 500

CHROMOSOMES = ['chr%s' % x for x in range(1, 23)]
CHROMOSOMES.extend(['chrX', 'chrY', 'chrM'])
CHROMSOME2CODE = {item: i + 1 for i, item in enumerate(CHROMOSOMES)}


def get_xpos(chrom, pos):
    if not chrom.startswith('chr'):
        chrom = 'chr{}'.format(chrom)
    return CHROMSOME2CODE[chrom] * int(1e9) + int(pos)


class Region():
    def __init__(self, chrom, pos, padding=500):
        self.chrom = chrom
        self.start = max(0, int(pos) - int(padding))
        self.end = (int(pos) + int(padding))
        self.xpos = get_xpos(self.chrom, self.start)


def ShowRegionList(RegionList, num):
    print "%d of Region in List:"
    for i in range(num):
        print RegionList[i].chrom, RegionList[i].start
    print "+" * 50

# Mearge the Region that has overlapping


def MergeRegionList(RegionList):
    s_time = time.time()
    print "Sorting Region List..."
    RegionList.sort(key=lambda x: x.xpos)
    ShowRegionList(RegionList, 100)
    print "Merging Region List..."
    Res = []  # New Region List, Ele in it should have no overlapping
    # Current Region that ready to add in Res. It will be add to Res if next
    # Region don't have overlap with it.
    Curr = RegionList[0]
    for Region in RegionList:
        if Curr.chrom == Region.chrom:  # Mearge Region has overlapping and on same chrom
            if Curr.end < Region.start:  # No Overlapping
                Res.append(Curr)
                Curr = Region
            else:  # Curr and Region has overlapping
                Curr.end = Region.end
        else:  # Start a new chrom, push the Curr
            Res.append(Curr)
            Curr = Region
    Res.append(Curr)
    print "Merging Region Complete. Used %.3f s" % (time.time() - s_time)
    return Res


def Write(RegionList, OutName):
    s_time = time.time()
    print 'Writing results into file %s' % OutName
    fout = open(OutName, 'wb')
    for Region in RegionList:
        record = '\t'.join([str(Region.chrom), str(
            Region.start), str(Region.end)]) + '\n'
        fout.write(record)
    print 'Wrrting File Complete, Used %.3f s' % (time.time() - s_time)


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gatk', type=str, help='GATK Call Set.')
    parser.add_argument('-u', '--union', type=str,
                        help='Union of All Other Call Set.')
    parser.add_argument('-o', '--outname', type=str,
                        default='LocalAssemblyRegions.txt', help='Union of All Other Call Set.')
    args = parser.parse_args()
    return args.gatk, args.union, args.outname


def GetHand(filename):
    if filename.endswith('.vcf.gz'):
        return gzip.open(filename, 'rb')
    elif filename.endswith('.vcf'):
        return open(filename, 'rb')
    else:
        print "Error with file name, must be .vcf or .vcf.gz"
        exit()


def OutPutRegion(gatk):
    stime = time.time()
    print 'Reading Region contain Indels from %s' % gatk
    Gatk_hand = GetHand(gatk)
    RegionList = []
    counter = 0
    for l in Gatk_hand:

        if l .startswith('#'):
            continue
        counter += 1
        Chrom, Pos, ID, Ref, Alts = l.strip().split('\t')[0:5]
        Alts = Alts.split(',')
        for Alt in Alts:
            if len(Ref) != len(Alt):
                RegionList.append(Region(Chrom, Pos, PADDING_SIZE))
                break
        if counter % 1e11:
            print 'Read 100G records'
    print "Reading Region Completed, Used %.3f s" % (time.time() - stime)
    return RegionList


def main():
    gatk, union, OutName = GetOptions()
    RegionList = OutPutRegion(gatk)
    RegionList = MergeRegionList(RegionList)
    Write(RegionList, OutName)
    return


if __name__ == '__main__':
    main()
