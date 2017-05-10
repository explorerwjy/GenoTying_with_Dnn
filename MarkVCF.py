#!/home/yufengshen/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# PrepareTrainingData.py:
# Use vcf file as input, Generate window for each candidate variant.
#
# If use Multi-processing, will produce splited data and then need to merge them back.
#=========================================================================

import argparse
from utils import *
import gzip
import time
import os
import sys

sys.stdout = sys.stderr

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--vcf',
        type=str,
        required=True,
        help='Candidate variants to scan')
    parser.add_argument(
        '-t',
        '--true',
        type=str,
        required=True,
        help='True Positive Variant VCF. All candidate regions will be labeled according to this file')

    args = parser.parse_args()

    return args.vcf, args.true


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
            k, p, v = var2kv2(l)
            if k not in res:
                res[k] = v
            else:
                print "Multiple record in %s has same position: %s" % (T_vcf, p)
    return res

def VCFopen(vcf):
    if vcf.endswith('.vcf.gz'):
        return gzip.open(vcf)
    else:
        return open(vcf)

def VarScan(vcf, Positives, outName):
    hand = VCFopen(vcf)
    fout = open(outName, 'wb')
    for l in hand:
        if l.startswith('#'):
            fout.write(l)
            continue
        else:
            var = Variant(l)
            k = get_xpos(var.Chrom, var.Pos)
            l = Positives.get(k, None)
            if l == None: # this position don't has var
                var.label = '0'
            else:
                var.label = get_Genotype(l.strip().split('\t'))
            fout.write(var.out())
    hand.close()
    fout.close()

class Variant:
    def __init__(self, record):
        record = record.strip().split('\t')
        self.Chrom, self.Pos, self.Id, self.Ref, self.Alt, self.Qual, self.Filter, self.Info_str, self.Format = record[0:9]
        self.Genotypes = record[9:]
        #self.Alts = self.Alt.split(',')
        #self.Alleles = [self.Ref] + self.Alts
        #self.GetInfo()

    def GetInfo(self):
        self.Info = {}
        tmp = self.Info_str.split(';')
        for kv in tmp:
            try:
                k, v = kv.split('=')
                if k not in self.Info:
                    self.Info[k] = v.split(',')
                else:
                    self.Info[k].extend(v.split(','))
            except:
                pass

    def out(self):
        INFO = '{};Label={}'.format(self.Info_str, self.label)
        return '\t'.join([self.Chrom, self.Pos, self.Id, self.Ref, self.Alt, self.Qual, self.Filter, INFO, self.Format, '\t'.join(self.Genotypes)]) + '\n'

    def GetGT(self):
        GT = re.findall('[\d.]', self.Genotypes[0].split(':')[0])
        if GT[0] == '0' and GT[1] == '0':
            return 0  # Homozygous Ref
        elif GT[0] != GT[1]:
            return 1  # Hetrozygous
        else:
            return 2  # Homozygous Alt

def main():
    vcf, T_vcf = GetOptions()
    Positives = Get_Positives(T_vcf)
    outName = 'Labeled.' + vcf.strip().split('/')[-1].rstrip('.gz')
    VarScan(vcf, Positives, outName)


if __name__ == '__main__':
    main()
