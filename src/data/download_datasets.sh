# run this file within the VALL-E/src/data/ folder
echo "Downloading LibriLight"
wget https://dl.fbaipublicfiles.com/librilight/data/small.tar
echo "Untar-ing LibriLight"
tar -xvf small.tar
echo "Downloading LibriSpeech"
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
echo "Untar-ing and gz-ing LibriSpeech"
tar -xzvf train-clean-100.tar.gz
echo "To download the free spoken digit dataset, please visit: https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd"
echo "After downloading, unzip the dataset in the VALL-E/src/data/ folder."
echo "Dataset Download and Extraction Complete"