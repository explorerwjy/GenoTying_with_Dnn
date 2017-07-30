#!/home/yufengshen/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# PrepareTrainingData.py:
# Use vcf file as input, Generate window for each candidate variant.
# Chr1 - Chr19 as training set, Chr20 - Chr22 as test set.
#
# If use Multi-processing, will produce splited data and then need to merge them back.
#=========================================================================

import argparse
import pysam
from utils import *
import Region
import gzip
import multiprocessing
import time
import os
import subprocess
import sys
import csv

sys.stdout = sys.stderr


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--ref',
        type=str,
        default='/home/yufengshen/resources/reference_genomes/hg19/hg19.fasta',
        help='Reference Genome')
    parser.add_argument('-b', '--bam', type=str, required=True, help='Aligned bam file')
    parser.add_argument('--bamout', type=str, help='Aligned bamout file, re-assembly result from GATK. If this option is specified, This script will first try to make up a Region from bamout bam, if no reads from bamout, it will make up Region from original bam.')
    parser.add_argument(
        '-v',
        '--vcf',
        type=str,
        required=True,
        help='Candidate variants to scan')
    parser.add_argument(
        '-m',
        '--mode',
        default=1,
        choices=[
            1,
            2],
        help='Mode. 1.VarScan(Scan through all variants from a vcf file) 2.OneVar (make up Region at a particular site)')
    parser.add_argument(
        '-t',
        '--true',
        type=str,
        required=True,
        help='True Positive Variant VCF. All candidate regions will be labeled according to this file')
    parser.add_argument('-o', '--output', help='Output Name for pile up windows for candidate genes')
    parser.add_argument(
        '-p',
        '--process',
        type=int,
        default=1,
        help='Num of process used to produce result')
    parser.add_argument('--split_training_testing', type=bool, default=False, help='Whether split data into training data (chr 1-19) and testing data (chr 20-22, X, Y)')
    args = parser.parse_args()
    if args.output == None:
        args.output = args.vcf.split('/')[-1].rstrip(".gz").rstrip(".vcf")
    return args

class PrepareTrainingData:
    def __init__(self, args):
        self.mode = args.mode 
        self.ReferenceGenome = args.ref
        self.Bam = args.bam
        self.Bamout = args.bamout
        self.VCF = args.vcf
        self.PositiveFil = args.true
        self.split_training_testing = args.split_training_testing
        self.output = args.output
        self.Nprocess = args.process
        print "Mode:",self.mode
        print "ReferenceGenome:",self.ReferenceGenome
        print "Bam:",self.Bam
        print "Bamout:",self.Bamout
        print "VCF:",self.VCF 
        print "Labels from:",self.PositiveFil
        print "split_training_testing:", self.split_training_testing
        print "OutName:", self.output
        print "Number of process:",self.Nprocess

    def run(self):
        s_time = time.time()
        if self.mode == '2':
            self.OneVar(self.bam)
        else:
            self.PositiveVar = self.Get_Positives()
            self.VarScan()
        print "Total Running Time is %.3f"%(time.time()-s_time)

    def Get_Positives(self):
        print "Start Loading True Variants at {}".format(self.PositiveFil)
        # READ FROM VCF
        if self.PositiveFil.endswith(".vcf.gz") or self.PositiveFil.endswith(".vcf"):
            if self.PositiveFil.endswith('.vcf.gz'):
                fin = gzip.open(self.PositiveFil)
            else:
                fin = open(self.PositiveFil)
            res = {}
            for l in fin:
                if l.startswith('#'):
                    continue
                else:
                    k, p, v = var2kv2(l)
                    if k not in res:
                        v = get_Genotype(l.strip().split('\t'))
                        res[k] = v
                    else:
                        print "Multiple record in %s has same position: %s" % (self.PositiveFil, p)
            print "Finish Load True Variants"
            return res
        else:
            if self.PositiveFil.endswith("csv"):
                reader = csv.reader(open(self.PositiveFil, 'rb'), delimiter=',')
            else:
                reader = csv.reader(open(self.PositiveFil, 'rb'), delimiter='\t')
            header = reader.next()
            header = map(lambda x:x.lstrip("#"), header)
            if (not 'Chrom' in header) or (not 'Position' in header) or (not 'Ref' in header) or (not 'Alt' in header) in header:
                print "Header Error in PositiveFil."
                exit()
            res = {}
            for row in reader:
                tmp_dict = dict(zip(header ,row))
                #key = tmp_dict('Chrom') + ":" + tmp_dict('Position')
                key = get_xpos(tmp_dict['Chrom'], tmp_dict['Position'])
                value = tmp_dict['Ref'] + ":" +tmp_dict['Alt']
                try:
                    res[key] = tmp_dict['GT']
                except:
                    res[key] = '1'
            return res

    # Scan a candidate vcf file, generate window for the variant and mark
    # genotype according to GIAB positives
    def VarScan(self):
        jobs = []
        for i in range(self.Nprocess):
            p = multiprocessing.Process(
                target=self.load_variants,
                args=( i,))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
        # Merge all files
        if self.split_training_testing:
            print "Merging Files together and bgzip"
            if not os.path.exists('./tmp.sort.{}.training'.format(self.output)):
                os.makedirs('./tmp.sort.{}.training'.format(self.output))
            if not os.path.exists('./tmp.sort.{}.testing'.format(self.output)):
                os.makedirs('./tmp.sort.{}.testing'.format(self.output))
            command1 = 'cat tmp.{}.train.*.GtdRegion.txt| sort -k1,1d -k2,2n -T ./tmp.sort.{}.training > {}.Training.GtdRegion.txt ;bgzip -f {}.Training.GtdRegion.txt; tabix -f -s 1 -b 2 -e 3 {}.Training.GtdRegion.txt.gz'.format(self.output,self.output,self.output,self.output,self.output)
            command2 = 'cat tmp.{}.test.*.GtdRegion.txt| sort -k1,1d -k2,2n -T ./tmp.sort.{}.testing > {}.Testing.GtdRegion.txt ;bgzip -f {}.Testing.GtdRegion.txt; tabix -f -s 1 -b 2 -e 3 {}.Testing.GtdRegion.txt.gz'.format(self.output,self.output,self.output,self.output,self.output)
            process1 = subprocess.Popen(command1, shell=True, stdout=subprocess.PIPE)
            process2 = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE)
            process1.wait()
            process2.wait()
            print process1.returncode
            print process2.returncode
            print "Clean Up Tmp Files"
            process3 = subprocess.Popen('rm -rf ./tmp.sort.{}.* tmp.{}.*.GtdRegion.txt'.format(self.output, self.output), shell=True, stdout=subprocess.PIPE)
            process3.wait()
            print process3.returncode
            print "Done!"
        else:
            print "Merging Files together and bgzip"
            if not os.path.exists('./tmp.sort.{}'.format(self.output)):
                os.makedirs('./tmp.sort.{}'.format(self.output))
            command1 = 'cat tmp.{}.*.GtdRegion.txt| sort -k1,1d -k2,2n -T ./tmp.sort.{} > {}.GtdRegion.txt ;bgzip -f {}.GtdRegion.txt; tabix -f -s 1 -b 2 -e 3 {}.GtdRegion.txt.gz'.format(self.output, self.output, self.output, self.output,self.output)
            process1 = subprocess.Popen(command1, shell=True, stdout=subprocess.PIPE)
            process1.wait()
            print process1.returncode
            print "Clean Up Tmp Files"
            process3 = subprocess.Popen('rm -rf ./tmp.sort.{} tmp.{}.*.GtdRegion.txt'.format(self.output, self.output), shell=True, stdout=subprocess.PIPE)
            process3.wait()
            print process3.returncode
            print "Done!"


    def load_variants(self, i):
        window_generator = self.parse_tabix_file_subset(i, self.get_variants_from_sites_vcf)
        if self.split_training_testing:
            outname_train = 'tmp.{}.train.'.format(self.output) + str(i) + '.GtdRegion.txt'
            outname_test = 'tmp.{}.test.'.format(self.output) + str(i) + '.GtdRegion.txt'
            fout_train = open(outname_train, 'wb')
            fout_test = open(outname_test, 'wb')
            for record in window_generator:
                if record.chrom not in ['20', '21', '22', 'X', 'Y']:
                    fout_train.write(record.write())
                elif record.chrom in ['20', '21', '22', 'X', 'Y']:
                    fout_test.write(record.write())
        else:
            outname = 'tmp.{}.'.format(self.output) + str(i) + '.GtdRegion.txt'
            fout = open(outname, 'wb')
            for record in window_generator:
                fout.write(record.write())

    def parse_tabix_file_subset(self, subset_i, record_parser):
        start_time = time.time()
        tabix_filenames = [self.VCF]
        open_tabix_files = [pysam.Tabixfile(tabix_filename) for tabix_filename in tabix_filenames]
        tabix_file_contig_pairs = [
            (tabix_file, contig) for tabix_file in open_tabix_files for contig in tabix_file.contigs]
        tabix_file_contig_subset = tabix_file_contig_pairs[subset_i:: self.Nprocess]
        short_filenames = ",".join(map(os.path.basename, tabix_filenames))
        num_file_contig_pairs = len(tabix_file_contig_subset)
        print "Lodaing subset %d from %d" % (subset_i, self.Nprocess)

        RefFile = pysam.FastaFile(self.ReferenceGenome)
        SamFile = pysam.AlignmentFile(self.Bam, "rb")
        if self.Bamout != None:
            BamoutFile = pysam.AlignmentFile(self.Bamout, "rb")
        else:
            BamoutFile = None

        counter = 0
        for tabix_file, contig, in tabix_file_contig_subset:
            #header_iterator = tabix_file.header
            records_iterator = tabix_file.fetch(
                contig, 0, 10**9, multiple_iterators=True)
            # for parsed_record in record_parser(itertools.chain(header_iterator,
            # records_iterator), Positive_vars, RefFile, SamFile ):
            for parsed_record in record_parser(
                    records_iterator, RefFile, SamFile, BamoutFile):
                counter += 1
                yield parsed_record

                if counter % 10000 == 0:
                    seconds_elapsed = float(time.time() - start_time)
                    print "Load %d records from subset %d of %d from %s in %f seconds" % (counter, subset_i, subset_n, short_filenames, seconds_elapsed)

    # The record_parser in parse_tabix_file_subset


    def get_variants_from_sites_vcf(self, sites_file, RefFile, SamFile, BamoutFile):
        for l in sites_file:
            # if l.startswith('##'):
            #	continue
            # elif l.startswith('#'):
            #	continue
            llist = l.strip().split('\t')
            k, chrom, pos, ref, alt = var2kv(llist)
            if k in self.PositiveVar:
                try:
                    GT = self.PositiveVar[k]
                except:
                    continue
                region = Region.CreateRegion(
                    RefFile,
                    SamFile,
                    BamoutFile,
                    chrom,
                    pos,
                    ref,
                    alt,
                    str(GT))  # Create a Region according to a site
            else:
                region = Region.CreateRegion(
                    RefFile, SamFile, BamoutFile, chrom, pos, ref, alt, '0')
            yield region


    # This func used to view window in a given region. Mainly aimed to debug
    # the Region part.
    def OneVar(self):
        RefFile = pysam.FastaFile(ref)
        SamFile = pysam.AlignmentFile(bam, "rb")
        while True:
            tmp = raw_input('Please enter the chr:pos >> ').split(':')
            if len(tmp) == 2:
                chrom, pos = tmp
                # Create a Region according to a site
                region = Region.CreateRegion(
                    RefFile, SamFile, chrom, pos, 'N', True)
                region.show()
            elif tmp[0] == 'N' or tmp[0] == 'n':
                exit()
            else:
                print tmp
                print tmp.split(':')
                exit()


    def Pulse(self, region):
        region.show()
        ans = raw_input("Go to next var (y/n)? >>")
        if ans.lower() == 'y':
            return
        else:
            exit()


def main():
    args = GetOptions()
    ins = PrepareTrainingData(args)
    ins.run()


if __name__ == '__main__':
    main()
