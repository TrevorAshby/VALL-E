# Datasets

## LibriSpeech ASR corpus
This dataset can be found <a href="http://www.openslr.org/12">here</a>. For this project, I am going to download the train-clean-100.tar.gz [6.3GB]. 

## LibriLight
This dataset can be found <a href="https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md">here</a>. For this project, I am going to download the small.tar [35GB] version. Small.tar contains 577 hours of audio, for a total of 35GB.

- md5: `c49207eb86a8e8ac895561c37232041e`

The directory structure of LibriSpeech:

    dataset_name/speakerID/book_name/

Where `dataset_name` is `small`, `medium`, `large`. `speakerID` is the LibriVox speakerID (number), and `book_name` is the name of the original LibriVox audiobook file.