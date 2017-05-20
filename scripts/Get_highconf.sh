VCF=$1
BED=$2
Nam=$(basename $VCF| sed s/.gz//g| sed s/.vcf//g)
OutVCF=$Nam.highconf

vcftools --gzvcf $VCF --bed $BED --recode --recode-INFO-all --out $OutVCF


