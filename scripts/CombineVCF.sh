GATKJAR=/home/local/users/jw/bin/GenomeAnalysisTK.jar
RefGenome=/home/local/users/jw/resources/reference_genomes/hg19/hg19.fasta
while getopts g:s:p:f:m:o opt; do
	case "$opt" in
		g) GATK="$OPTARG";;
		s) ST="$OPTARG";;
		p) PT="$OPTARG";;
		f) FB="$OPTARG";;
		m) MIN="$OPTARG";;
		o) OutFil="CombinedVCF.vcf";;
	esac
done
TmpDir="CombineVCF.Tmp.dir"; mkdir -p $TmpDir
OutFil="CombinedVCF.vcf"
java -Xmx12G -Djava.io.tmpdir=$TmpDir -jar $GATKJAR \
	-T CombineVariants \
	-R $RefGenome \
	--variant:gatk $GATK \
	--variant:st $ST \
	--variant:pt $PT \
	--variant:fb $FB \
	-o Raw.$OutFil \
	-genotypeMergeOptions PRIORITIZE \
	-priority gatk,pt,st,fb \
	-minN $MIN;

java -Xmx4G -XX:ParallelGCThreads=1 -Djava.io.tmpdir=$TmpDir -jar $GATKJAR \
	-T LeftAlignAndTrimVariants \
	-R $RefGenome \
	-V Raw.$OutFil \
	-o $OutFil
 
rm -rf $TmpDir Raw.$OutFil*;
echo "ConbineVCF Done!";

