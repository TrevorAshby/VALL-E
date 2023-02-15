# VALL-E
CS 674 Project 1: My plan is to replicate the work done to create VALL-E


Paper link: https://arxiv.org/pdf/2301.02111.pdf

Could be worth looking at this link down the road: https://github.com/enhuiz/vall-e

|Day|Hours|Accomplished|
|---|---|---|
|2/14|7:00PM - 8:15PM = 1.25hrs|Read paper, wrote notes and README
|2/14|9:00PM - 10:15PM = 1.25hrs|Wrote script to download datasets
|2/14|10:15PM - 1:00AM = 2.75hrs|Learned about DNN-HMM models, and began writing one for phoneme conversion
|**Total**|**5.25**|

Architecture:
<img src="valle.png" width=700px>

## Phoneme Conversion
"Hybrid DNN-HMM ASR model on 960 hours labeled LibriSpeech following Kaldi recipe. Once trained, unlabeled speech data is decoded and transduced to best phoneme-level alightment paths where frameshift is 30ms"

<img src="image (2).png" width=700px>

This is a link to a DNN-HMM ASR paper: https://ieeexplore.ieee.org/abstract/document/6681449
Based upon my research, these are also referred to as "Chain" models
Another helpful link: https://github.com/raminnakhli/HMM-DNN-Speech-Recognition

### Entire list of all 44 phonemes in the English Language
<img src="image.png" width=700px>
<img src="image (1).png" width=700px>

## Audio Codec Encoder
Link: https://arxiv.org/pdf/2209.03143.pdf
<img src="neural_audio_codec.png" width=700px>

## Neural Codec Language Modeling
Transformer architecture with 12 layers, 16 attention heads, embedding dimension of 1024, a feed-forward layer dimension of 4096, and a dropout of 0.1. Average length of waveform in LibriLight is 60 seconds. During training, randomly crop the waveform to a random length between 10 seconds and 20 seconds. It's corresponding phoneme alignments are used as the phoneme prompt. 

AdamW optimizer.


### Encodec
They use a pre-trained one, "EnCodec" 

"EnCodec is a convolutional encoder-decoder model, whose input and output are both 24kHz audio across variable bitrates."

EnCodec Links: 
    - https://arxiv.org/pdf/2210.13438.pdf
    - https://pypi.org/project/encodec/
    - https://github.com/facebookresearch/encodec

## Audio Codec Decoder

## Datasets
LibriLight, 60K hours of English speech, with over 7000 unique speakers. Original data is audio-only, so ASR is used to generate transcriptions.

Link: https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md

    LibriSpeech 
Link: http://www.openslr.org/12

<br>

## TO DO
1. ~~Write a script to download the datasets~~
2. Figure out DNN-HMM ASR model for transcribing, then convert to phonemes
3. install and setup EnCodec
4. Build Transformer Decoder
5. Setup NAR and AR configurations for Decoder