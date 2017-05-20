NUM_Training=`gunzip -c ./Training.windows.txt.gz| wc -l`
NUM_Testing=`gunzip -c ./Testing.windows.txt.gz| wc -l`
OUT="DataSummary"

echo "TrainingData: $NUM_Training records" > $OUT
echo "TestingData: $NUM_Testing records" >> $OUT
