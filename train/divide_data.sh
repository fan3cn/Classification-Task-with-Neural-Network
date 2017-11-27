rm TrainData.csv CrossValidationData.csv
shuf extracted_features.csv  > train_data_shuf.csv
head -3500 train_data_shuf.csv > TrainData.csv
tail -1550 train_data_shuf.csv > CrossValidationData.csv
rm train_data_shuf.csv
