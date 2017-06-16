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
    #parser.add_argument('--bamoutRegion', type=str, help='Aligned bamout file, re-assembly result from GATK')
    parser.add_argument(
        '-v',
        '--vcf',
        type=str,
        required=True,
        help='Candidate variants to scan')
    parser.add_argument(
        '-p',
        '--process',
        type=int,
        default=1,
        help='Num of process used to produce result')
    parser.add_argument('-o', '--outname', type=str, help="Output Name of the Output Tensor File.")
    args = parser.parse_args()
    if args.outname == None:
        args.outname = args.vcf.rstrip('.gz').rstrip('.vcf') + '.CandidateRegions.txt'
    return args.ref, args.bam, args.bamout, args.vcf, args.outname, args.process

def VarScan(referenceGenome, bam, bamout, Candidate_vcf, OutName, Nprocess):
    jobs = []
    for i in range(Nprocess):
        p = multiprocessing.Process(
            target=load_variants,
            args=(
                Candidate_vcf,
                referenceGenome,
                bam,
                bamout,
                i,
                Nprocess))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    # Merge all files
    print "Merging Files together and bgzip"
    if not os.path.exists('./tmp_sort_{}'.format(OutName)):
    command1 = 'cat tmp.*.{}| sort -k1,1d -k2,2n -T ./tmp_sort_{} > {} ;bgzip -f {}; tabix -f -s 1 -b 2 -e 3 {}.gz'.format(OutName, OutName, OutName, OutName, OutName)
    process1 = subprocess.Popen(command1, shell=True, stdout=subprocess.PIPE)
    process1.wait()
    print process1.returncode
    #process3 = subprocess.Popen('rm tmp.*.windows.txt', shell=True, stdout=subprocess.PIPE)
    # process3.wait()
    # print process3.returncode
    print "Done!"


def load_variants(VCF, referenceGenome, bam, bamout, OutName, i, n):
    outname = 'tmp.' + str(i) + OutName
    fout = open(outname, 'wb')
    window_generator = parse_tabix_file_subset(
        [VCF],
        referenceGenome,
        bam,
        bamout,
        i,
        n,
        get_variants_from_sites_vcf)
    for record in window_generator:
        fout.write(record.write())


def parse_tabix_file_subset(
        tabix_filenames,
        referenceGenome,
        bam,
        bamout,
        subset_i,
        subset_n,
        record_parser):
    start_time = time.time()
    open_tabix_files = [pysam.Tabixfile(tabix_filename)
                        for tabix_filename in tabix_filenames]
    tabix_file_contig_pairs = [
        (tabix_file,
         contig) for tabix_file in open_tabix_files for contig in tabix_file.contigs]
    tabix_file_contig_subset = tabix_file_contig_pairs[subset_i:: subset_n]
    short_filenames = ",".join(map(os.path.basename, tabix_filenames))
    num_file_contig_pairs = len(tabix_file_contig_subset)
    print "Lodaing subset %d from %d" % (subset_i, subset_n)

    RefFile = pysam.FastaFile(referenceGenome)
    SamFile = pysam.AlignmentFile(bam, "rb")
    if bamout != None:
        BamoutFile = pysam.AlignmentFile(bamout, "rb")
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
                records_iterator, Positive_vars, RefFile, SamFile, BamoutFile):
            counter += 1
            yield parsed_record

            if counter % 10000 == 0:
                seconds_elapsed = float(time.time() - start_time)
                print "Load %d records from subset %d of %d from %s in %f seconds" % (counter, subset_i, subset_n, short_filenames, seconds_elapsed)

# The record_parser in parse_tabix_file_subset


def get_variants_from_sites_vcf(sites_file, RefFile, SamFile, BamoutFile):
    for l in sites_file:
        # if l.startswith('##'):
        #	continue
        # elif l.startswith('#'):
        #	continue
        llist = l.strip().split('\t')
        region = Region.CreateRegion(
                RefFile,
                SamFile,
                BamoutFile,
                chrom,
                pos,
                ref,
                alt)
        yield region

def main():
    s_time = time.time()
    referenceGenome, bam, bamout, vcf, OutName, Nprocess = GetOptions()
    VarScan(referenceGenome, bam, bamout, vcf, OutName, Nprocess)
    print "Total Running Time is %.3f"%(time.time()-s_time)

if __name__ == '__main__':
    main()
