import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from phoneme_conversion.pcm import PhonemeConversionModel

# 24khz version of VALL-E
class VALLE():
    def __init__(self):
        self.audio_codec = EncodecModel.encodec_model_24khz() # audio encoder/decoder - pretrained version from Facebook
        self.phoneme_converter = PhonemeConversionModel() # text to phoneme converter - in house DNN-HMM model (85% acc)
        self.NAR = None # non-autoregressive transformer decoder
        self.AR = None # autoregressive transformer decoder
        
    def forward(txt, wav):
        # txt -> phoneme
        
        # wav(3sec) -> encoding
        # wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0) # FIXME get rid of .unsqueeze once dealing with batches

        # phon+enc -> LLM

        # logits -> decoder

        # return personalized speech
        pass