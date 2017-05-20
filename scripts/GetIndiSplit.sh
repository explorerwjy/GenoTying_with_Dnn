InpFil=$1
for INDI in HG002 HG003 HG004
do
	echo $INDI
	nohup vcftools --recode --recode-INFO-all --indv $INDI --out $INDI --gzvcf $InpFil &

done

