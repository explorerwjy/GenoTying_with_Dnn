#===============================================================================
# Post processing for variants from different callers.
# GATK, Platypus, Samtools
# GATK should have been joint genotyping, merged and VQSR
# Other should have been merged.
# InpName: callername.vcf.gz
#===============================================================================
InpFil=$1
Ref=/home/local/ARCS/hz2408/pipeline/Exome_pipeline_scripts_GATKv3/WES_Pipeline_References.b37.sh
hg19=~/resources/reference_genomes/hg19/hg19.fasta
CodingFilter=/home/local/users/jw/Consensus_Pipeline/scripts/Filters.py
basename="${InpFil%.*.*}"
AnnotateVCF=/home/local/ARCS/hz2408/pipeline/Exome_pipeline_scripts_GATKv3/ExmVC.3.AnnotateVCF.sh
Split=/home/local/users/jw/Consensus_Pipeline/scripts/seperate_SNV_INDEL.pl

#===============================================================================
# Normlize the variants
#===============================================================================
echo "Normliza variants for $basename"
Norm="$basename.norm.vcf"
if [ ! -f $Norm ]; then
	vt normalize $InpFil -r $hg19 -o $Norm 2>stderr_$basename.norm.txt
else
	echo "$Norm Already exist"
fi

#===============================================================================
# Normlize the variants
#===============================================================================
echo "Decompose clumped variant for $basename"
NormDecom="$basename.decom.norm.vcf"
if [ ! -f $NormDecom ]; then
	vt decompose_blocksub -p $Norm -o $NormDecom 2>>stderr_$basename.norm.txt
else
	echo "$NormDecom Already exist"
fi
#===============================================================================
# Annotation the variants
#===============================================================================
echo "Annotation with Annovar"
Anno="$basename.decom.norm.annotated.vcf.gz"
if [ ! -f $Anno ]; then
	$AnnotateVCF -i $NormDecom -r $Ref
else
	echo "$Anno Already exist"
fi

#===============================================================================
# Filter the variants by VarClass and ExACfreq
#===============================================================================
#echo "Filter on the coding region and AF < 0.0001"
#Filter="$basename.filterd.vcf"
#if [ ! -f $Filter ]; then
#	python $CodingFilter -i $Anno -o $Filter > "Error_variants.$basename.txt"
#else
#	echo "$Filter Already exist"
#fi

#===============================================================================
# Split into SNV and INDEL
#===============================================================================
#echo Split into SNV INDEL
#SNP="$basename.snv.vcf"
#INDEL="$basename.indel.vcf"
#$Split $Filter $SNP $INDEL

echo Done
