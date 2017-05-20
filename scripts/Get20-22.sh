VCF=$1
Nam=$(basename $VCF| sed s/.gz//g)
OutVCF=20-22.$Nam

Header=$VCF.TMPHeader
TMP20=$VCF.TMP20
TMP21=$VCF.TMP21
TMP22=$VCF.TMP22

tabix -H $VCF > $Header
tabix $VCF 20 > $TMP20
tabix $VCF 21 > $TMP21
tabix $VCF 22 > $TMP22

cat $Header $TMP20 $TMP21 $TMP22 > $OutVCF
rm $Header $TMP20 $TMP21 $TMP22 
