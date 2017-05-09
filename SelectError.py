#!/home/yufengshen/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# SelectError.py
#=========================================================================

import argparse


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf', type=str, help='VCF file')
    args = parser.parse_args()

    return args.vcf

def ErrorRecord(l):
    llist = l.split('\t')
    label = llist[7].split('=')[1]
    gt = llist[9].split(':')[0]
    gt = sum( map(int, gt.split('/')) )
    if int(label) != int(gt):
        print l
        print int(label), int(gt)
        return False
    else:
        return True

def main():
    vcf = GetOptions()
    fin = open(vcf, 'rb')
    fout = open("Errors.vcf",'wb')
    for l in fin:
        if l.startswith('#'):
            fout.write(l)
        else:
            if not ErrorRecord(l):
                fout.write(l)

    return


if __name__ == '__main__':
    main()

