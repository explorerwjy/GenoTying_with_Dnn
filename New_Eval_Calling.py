#!/home/yufengshen/anaconda2/bin/python
import argparse
import gzip
import re
import time
from utils import *


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf',
                        help='VCF file to be eval. This VCF should have INFO:Label=(0|1|2)')
    parser.add_argument("-d", "--outDetail", action='store_true', default=False,
                        help="continue training from a checkpoint")
    args = parser.parse_args()
    return args.vcf, args.outDetail


class Counts():
    def __init__(self):
        self.zero_zero = 0
        self.zero_one = 0
        self.zero_two = 0
        self.one_zero = 0
        self.one_one = 0
        self.one_two = 0
        self.two_zero = 0
        self.two_one = 0
        self.two_two = 0

    def Get_POS_Eval(self):
        print self.zero_zero, self.zero_one, self.zero_two, self.one_zero,  self.one_one, self.one_two, self.two_zero, self.two_one, self.two_two
        self.All = self.zero_zero + self.one_zero + self.two_zero + self.zero_one + \
            self.zero_two + self.one_one + self.two_two + self.one_two + self.two_one
        self.Positive = self.zero_one + self.zero_two + \
            self.one_one + self.one_two + self.two_one + self.two_two
        self.POS_TP = self.one_one + self.one_two + self.two_one + self.two_two
        self.POS_FP = self.one_zero + self.two_zero
        self.POS_FN = self.zero_one + self.zero_two
        self.POS_SE = float(self.POS_TP) / (self.POS_TP + self.POS_FN)
        # self.POS_SE = float(self.POS_TP) / (self.Positive)
        self.POS_PPV = float(self.POS_TP) / (self.POS_TP + self.POS_FP)
        self.POS_F1 = float(2 * self.POS_TP) / \
            (2 * self.POS_TP + self.POS_FN + self.POS_FP)

    def Get_Genotype_Eval(self):
        self.GT_TP = self.one_one + self.two_two
        self.GT_FP = self.one_two + self.two_one + self.one_zero + self.two_zero
        self.GT_FN = self.zero_one + self.zero_two
        # self.GT_SE = float(self.GT_TP)/(self.GT_TP + self.GT_FN)
        self.GT_SE = float(self.GT_TP) / (self.Positive)
        self.GT_PPV = float(self.GT_TP) / (self.GT_TP + self.GT_FP)
        self.GT_F1 = float(2 * self.GT_TP) / \
            (2 * self.GT_TP + self.GT_FN + self.GT_FP)

    def show(self):
        print 'Eval Results on TestSet -> GroundTruth'
        print '0/0 -> 0/0:', self.zero_zero
        print '0/0 -> 0/1:', self.zero_one
        print '0/0 -> 1/1:', self.zero_two
        print '0/1 -> 0/0:', self.one_zero
        print '0/1 -> 0/1:', self.one_one
        print '0/1 -> 1/1:', self.one_two
        print '1/1 -> 0/0:', self.two_zero
        print '1/1 -> 0/1:', self.two_one
        print '1/1 -> 1/1:', self.two_two
        print
        print 'confusion matrix'
        print '%12d\t%12s\t%12s\t%12s' % (self.All,         'Predicted NO',   'Predicted YES', '')
        print '%12s\t%12d\t%12d\t%12d' % ('Actual NO', (self.zero_zero), (self.zero_one + self.zero_two), (self.zero_zero + self.zero_one + self.zero_two))
        print '%12s\t%12d\t%12d\t%12d' % ('Actual YES', (self.one_zero + self.two_zero), (self.one_one + self.two_two + self.one_two + self.two_one), (self.one_zero + self.two_zero + self.one_one + self.two_two + self.one_two + self.two_one))
        print '%12s\t%12d\t%12d\t%12s' % ('', (self.zero_zero + self.one_zero + self.two_zero), (self.zero_one + self.zero_two + self.one_one + self.two_two + self.one_two + self.two_one), '')
        print
        print '%12d\t%12s\t%12s\t%12s\t%12s' % (self.All, 'Predicted 0', 'Predicted 1', 'predicted 2', '')
        print '%12s\t%12d\t%12d\t%12d\t%12d' % ('Actual 0', self.zero_zero, self.one_zero, self.two_zero, (self.zero_zero + self.one_zero + self.two_zero))
        print '%12s\t%12d\t%12d\t%12d\t%12d' % ('Actual 1', self.zero_one, self.one_one, self.two_one, (self.zero_one + self.one_one + self.two_one))
        print '%12s\t%12d\t%12d\t%12d\t%12d' % ('Actual 2', self.zero_two, self.one_two, self.two_two, (self.zero_two + self.one_two + self.two_two))
        print '%12s\t%12d\t%12d\t%12s\t%12s' % ('', (self.zero_zero + self.zero_one + self.zero_two), (self.one_zero + self.one_one + self.one_two), (self.two_zero + self.two_one + self.two_two), '')
        print ''
        print '-' * 50
        print 'Position Eval:'
        print 'TP:', self.POS_TP
        print 'FP:', self.POS_FP
        print 'FN:', self.POS_FN
        print 'SE:', self.POS_SE
        print 'PPV:', self.POS_PPV
        print 'F1:', self.POS_F1
        print '-' * 50
        print 'Genotype Eval:'
        print 'TP:', self.GT_TP
        print 'FP:', self.GT_FP
        print 'FN:', self.GT_FN
        print 'SE:', self.GT_SE
        print 'PPV:', self.GT_PPV
        print 'F1:', self.GT_F1


class EvalCalling:
    def __init__(self, VCF, outDetail=False):
        self.VCF = VCF
        self.VCFhand = self.GetVCF()
        self.outDetail = outDetail
        if outDetail:
            self.fout2 = open('EvalCalling.vcf', 'wb')

    def GetVCF(self):
        if self.VCF.endswith('.vcf.gz'):
            return gzip.open(self.VCF)
        else:
            return open(self.VCF)

    def run(self):
        counts = Counts()
        for l in self.VCFhand:
            if l.startswith('#'):
                continue
            else:
                var = Variant(l)
                label = var.Info['Label'][0]
                GT = var.GetGT()
                var.eval = self.MarkError(str(label), str(GT), counts)
                if var.Markerror() and self.outDetail:
                    self.fout2.write(var.out())
        counts.Get_POS_Eval()
        counts.Get_Genotype_Eval()
        counts.show()

    def MarkError(self, label, GT, counts):
        if label == '0' and GT == '0':
            counts.zero_zero += 1
            return "0-0"
        elif label == '0' and GT == '1':
            counts.zero_one += 1
            return "0-1"
        elif label == '0' and GT == '2':
            counts.zero_two += 1
            return "0-2"
        elif label == '1' and GT == '0':
            counts.one_zero += 1
            return "1-0"
        elif label == '1' and GT == '1':
            counts.one_one += 1
            return "1-1"
        elif label == '1' and GT == '2':
            counts.one_two += 1
            return "1-2"
        elif label == '2' and GT == '0':
            counts.two_zero += 1
            return "2-0"
        elif label == '2' and GT == '1':
            counts.two_one += 1
            return "2-1"
        elif label == '2' and GT == '2':
            counts.two_two += 1
            return "2-2"


class Variant:
    def __init__(self, record):
        record = record.strip().split('\t')
        self.Chrom, self.Pos, self.Id, self.Ref, self.Alt, self.Qual, self.Filter, self.Info_str, self.Format = record[
            0:9]
        self.Genotypes = record[9:]
        self.Alts = self.Alt.split(',')
        self.Alleles = [self.Ref] + self.Alts
        self.GetInfo()

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

    def GetGT(self):
        GT = re.findall('[\d.]', self.Genotypes[0].split(':')[0])
        if GT[0] == '0' and GT[1] == '0':
            return 0  # Homozygous Ref
        elif GT[0] != GT[1]:
            return 1  # Hetrozygous
        else:
            return 2  # Homozygous Alt

    def Markerror(self):
        if self.eval == '0-1':
            return True
        elif self.eval == '0-2':
            return True
        elif self.eval == '1-0':
            return True
        elif self.eval == '1-2':
            return True
        elif self.eval == '2-0':
            return True
        elif self.eval == '2-1':
            return True
        else:
            return False

    def out(self):
        INFO = '{};Eval={}'.format(self.Info_str, self.eval)
        return '\t'.join([self.Chrom, self.Pos, self.Id, self.Ref, self.Alt, self.Qual, self.Filter, INFO, self.Format, '\t'.join(self.Genotypes)]) + '\n'


def main():
    vcf, detail = GetOptions()
    evalcall = EvalCalling(vcf, detail)
    evalcall.run()

if __name__=='__main__':
    main()
