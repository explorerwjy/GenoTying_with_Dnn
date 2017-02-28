#===============================================================================
# Post processing for variants from different callers.
# GATK, Platypus, Samtools
# GATK should have been joint genotyping, merged and VQSR
# Other should have been merged.
# InpName: callername.vcf.gz
#===============================================================================
InpFil=$1 #Input VCF to be norm
hg19=~/resources/reference_genomes/hg19/hg19.fasta
basename="${InpFil%.*.*}"

#===============================================================================
# Normlize the variants
#===============================================================================
echo "Decompose clumped variant for $basename"
Decom="$basename.decom.vcf"
if [ ! -f $Decom ]; then
	vt decompose_blocksub -p $InpFil -o $Decom 2>stderr_$basename.norm.txt
else
	echo "$Decom Already exist"
fi

#===============================================================================
# Normlize the variants
#===============================================================================
echo "Normliza variants for $basename"
Norm="$basename.norm.vcf"
if [ ! -f $Norm ]; then
	vt normalize $Decom -r $hg19 -o $Norm 2>>stderr_$basename.norm.txt
else
	echo "$Norm Already exist"
fi

rm $Decom
echo Done
