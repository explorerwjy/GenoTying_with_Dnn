#!/home/yufengshen/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# Region
# A Region consists of Base Alignment, QualScore Alignemnt and Strand Alignment
# Output Format:
# Chrom Pos Pos Ref Alt FlatTensor
#=========================================================================

from optparse import OptionParser
import random
from utils import *
import gzip

#WIDTH = 31
WIDTH = 101
REF_LEN = 5
#HEIGHT = 50  # +REF_LEN Reference Seq
HEIGHT = 104  # +REF_LEN Reference Seq
DEPTH = 3
INSERT='I'
DELET='D'
INDEL_ANCHORING_BASE = 'X' # Not Used this version
#BASE = {'A': '1', 'T': '2', 'G': '3', 'C': '4', 'X': '5', 'N': '6'}
#BASE2 = {'0': '0', '1': 'A', '2': 'T', '3': 'G', '4': 'C', '5': 'X', '6': 'N'}
# -3 -2 -1  0  1  2  3
#  I  G  A  N  T  C  D
#  0  1  2  3  4  5  6
# For store conventient, plus 3
BASE = {'I': '0', 'G': '1', 'A': '2', 'N': '3', 'T': '4', 'C': '5', 'D': '6'}
BASE2 = {'0': 'I', '1': 'G', '2': 'A', '3': 'N', '4': 'T', '5': 'C', '6': 'D'}

#  Qual [ 0, 60]
# ASCII [33, 93] --> [!, ]]

# Strand
# -1  0  1
#  R  N  F
#  0  1  2
STRAND = {'F':'2', 'N':'1', 'R':'0'}

def get_full_cigar(cigartuples):
    res = []
    for (cigar_type, cigar_len) in cigartuples:
        res.extend([cigar_type] * cigar_len)
    return res


def per_base_alignment(start, end, pos, ref, read):
    # yield ref-pos, read_pos, cigar_char at each ref-read alignment
    # Pointer in ref  : p - start
    # Pointer in read : p + (read.reference_start - start)
    cigars = get_full_cigar(read.cigartuples)
    # print start,read.reference_start,end
    p_ref = max(read.reference_start, start)
    p_read = p_ref
    #p_read = p_read - cigars[:p_read - read.reference_start].count(1)
    p_cigar = p_ref
    # Adjust start position by insertion
    p_read = p_read - cigars[:p_cigar - read.reference_start].count(2)
    # Adjust start position by deletion
    p_read = p_read + cigars[:p_cigar - read.reference_start].count(1)
    #p_read = p_read - cigars[:p_cigar - read.reference_start].count(4)
    # print p_cigar, read.reference_start, cigars[:p_cigar -
    # read.reference_start].count(2)
    while True:
        if (p_ref >= end) or (p_read - read.reference_start >=
                              read.query_length):  # pointer exceed the region or read used up
            break
        elif cigars[p_cigar - read.reference_start] == 0 or cigars[p_cigar - read.reference_start] == 4 or cigars[p_cigar - read.reference_start] == 5:  # Match
            yield p_ref - start, p_read - read.reference_start, cigars[p_cigar - read.reference_start]
            p_ref += 1
            p_read += 1
            p_cigar += 1
        elif cigars[p_cigar - read.reference_start] == 1:  # Insertion
            yield p_ref - start, p_read - read.reference_start, cigars[p_cigar - read.reference_start]
            p_read += 1
            p_cigar += 1
        elif cigars[p_cigar - read.reference_start] == 2:  # Deletion
            yield p_ref - start, p_read - read.reference_start, cigars[p_cigar - read.reference_start]
            p_ref += 1
            p_cigar += 1
        else:
            print 'Unexpected CIGAR', cigars[p_cigar - read.reference_start]
            print cigars
            exit()


class Region():
    def __init__(self, chrom, pos, start, end, ref, alt, Y):
        self.chrom = chrom
        self.pos = pos
        self.start = start
        self.end = end
        self.ref = ref
        self.alt = alt
        self.label = Y
        self.base = []
        self.qual = []
        self.strand = []
        for raw in xrange(HEIGHT + 1):
            self.base.append(['3'] * (WIDTH))
            self.qual.append(['!'] * (WIDTH))
            self.strand.append(['1'] * (WIDTH))

    def show(self, all=False):
        print self.chrom + ':' + str(self.pos) + self.ref + self.alt + '\n'
        for row in self.base:
            print "#BASE"
            print ''.join(row)
        if all:
            print "#QUAL"
            for row in self.qual:
                print ''.join(row)
            print "#Strand"
            for row in self.strand:
                print ''.join(row)

    def write(self):
        # Align = self.label + Chrom2Byte(self.chrom) + Pos2Byte(self.pos) # 13
        # Byte Meta
        tmp = []
        for row in self.base:
            tmp.append(''.join(row))
        bases = ''.join(tmp)
        tmp = []
        for row in self.qual:
            tmp.append(''.join(row))
        quals = ''.join(tmp)
        tmp = []
        for row in self.strand:
            tmp.append(''.join(row))
        strands = ''.join(tmp)
        FlatTensor = ''.join([bases, quals, strands])
        return '\t'.join([self.chrom, str(self.pos), str(
            self.pos), self.ref, self.alt, self.label, FlatTensor]) + '\n'

    def fill_ref(self, ref):
        for row in xrange(REF_LEN):
            for col in xrange(WIDTH):
                # print row,col
                # print ref[col],len(ref)
                # print self.base[row],len(self.base[row])
                self.base[row][col] = BASE[ref[col]]  # Ref base
                self.qual[row][col] = get_qual(60)  # Ref are high qual
                self.strand[row][col] = get_strand(True)  # Ref are forword strand
        return row + 1

    def fill_read(self, ref, read, row_i):
        for ref_pos, read_pos, cigar_elt in per_base_alignment(
                self.start, self.end, self.pos, ref, read):
            read_base = None
            if cigar_elt == 1: #Insertion
                col = ref_pos - 1
                read_base = INSERT 
            elif cigar_elt == 2: #Deletion
                col = ref_pos
                read_base = DELET 
            elif cigar_elt == 0:
                col = ref_pos
                read_base = read.query_sequence[read_pos]

            if read_base:
                # print self.base[row_i][col]
                self.base[row_i][col] = BASE[read_base]
                self.qual[row_i][col] = min(get_qual(read.query_qualities[read_pos]), 
                                            get_qual(read.mapping_quality))
                self.strand[row_i][col] = get_strand(not read.is_reverse)
            # print col, ref_pos, read_pos, cigar_elt


def get_qual(qual):
    return chr(qual + 33)


def get_strand(flag_forward):
    if flag_forward:
        return STRAND['F']
    else:
        return STRAND['R']


def CreateRegion(RefFile, SamFile, BamoutFile, chrom, pos, ref, alt, Y, verbose=False):
    pos = int(pos)
    start = pos - (WIDTH - 1) / 2
    end = pos + (WIDTH - 1) / 2 + 1
    # print 'chr'+str(chrom)+':'+str(pos),start,end,start-end
    region = Region(chrom, pos, start, end, ref, alt, Y)
    ref = RefFile.fetch(chrom, start, end)
    row_i = region.fill_ref(ref)
    reads = get_overlapping(SamFile, BamoutFile, chrom, pos, start, end)
    good_reads, bad_reads, extra_reads = 0, 0, 0
    for read in reads:
        # print read
        if row_i < HEIGHT:
            region.fill_read(ref, read, row_i)
            row_i += 1
            good_reads += 1
        elif not is_usable_read(read):
            bad_reads += 1
        else:
            extra_reads += 1
    if verbose:
        print "%d Good reads, %d Bad reads, %d Extra reads." % (good_reads, bad_reads, extra_reads)
    return region

# Get Reads within region & Down Sampling reads


def get_overlapping(SamFile, BamoutFile, chrom, pos, start, end):
    narrow_start = pos - 1
    narrow_end = pos + 1
    if BamoutFile == None:
        raw_reads = SamFile.fetch(chrom, narrow_start, narrow_end)
    else:
        raw_reads = BamoutFile.fetch(chrom, narrow_start, narrow_end)
        if Getlen(raw_reads) == 0:
            raw_reads = SamFile.fetch(chrom, narrow_start, narrow_end)
    res = []
    for read in raw_reads:
        if is_usable_read(read):
            res.append(read)
    if len(res) > HEIGHT - REF_LEN:
        random.shuffle(res)
        res = res[:HEIGHT]
        res.sort(key=lambda x: x.reference_start, reverse=False)
    return res

def Getlen(raw_reads):
    res = 0
    for reads in raw_reads:
        res += 1
        if res > 0:
            return res
    return 0

def is_usable_read(read):
    return (
        not read.is_unmapped) and (
        not read.is_duplicate) and (
            'S' not in read.cigarstring) and (
                'H' not in read.cigarstring)


def GetOptions():
    parser = OptionParser()
    parser.add_option(
        '-i',
        '--input',
        dest='InpFil',
        metavar='InpFil',
        help='InpFile Read From')
    (options, args) = parser.parse_args()

    return options.InpFil

# Convert a line into readable window


def Line2Window_1(l):
    flag = l[:13]
    data = l[13:]
    print 'Label: %s\tChrom: %s\tPos: %s' % (flag[0], Byte2Chrom(flag[1:3]), Byte2Pos(flag[3:13]))
    start = 13
    height = 0
    depth = 0
    while depth < 3:
        if height > HEIGHT:
            depth += 1
            height = 0
        if depth == 0:
            print ''.join(map(lambda x: BASE2[x], list(l[start:start + WIDTH])))
        else:
            print l[start:start + WIDTH]
        start += WIDTH
        height += 1

def Line2Window(l):
    Chr, Start, End, Ref, Alt, Label, Data = l.strip().split('\t')
    print 'Label: %s\tChrom: %s\tPos: %s\t%s->%s' % (Label, Chr, End, Ref, Alt)
    start = 0
    height = 0
    depth = 0
    while depth < 3:
        if height > HEIGHT:
            depth += 1
            height = 0
        if depth == 0:
            print ''.join(map(lambda x: BASE2[x], list(Data[start:start + WIDTH])))
        else:
            print Data[start:start + WIDTH]
        start += WIDTH
        height += 1

def Dump(InpFil):
    if InpFil.endswith('.gz'):
        fin = gzip.open(InpFil)
    else:
        fin = open(InpFil)

    for l in fin:
        Line2Window(l)
    fin.close()


def main():
    InpFil = GetOptions()
    Dump(InpFil)
    return


if __name__ == '__main__':
    main()
