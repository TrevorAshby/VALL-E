# The phoneme conversion model, implemented to convert either text or audio to phonemes
# This is done by using the g2p package, and the whisper model 
import argparse
import torch
import torchaudio
import transformers
from g2p_en import G2p
import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class PhonemeConversionModel():
    def __init__(self):
        self.g2p = G2p()

        # load model and processor
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    def grapheme_to_phoneme(self, txt):
        return self.g2p(txt)

    def audio_to_phoneme(self, wav, sr):
        input_features = self.whisper_processor(wav, sampling_rate=sr, return_tensors="pt").input_features
        predicted_ids = self.whisper.generate(input_features, max_length=448)
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return transcription, self.grapheme_to_phoneme(transcription)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('user_input', nargs='?' ,type=str, default="This is the default.")
    parser.add_argument('-t', '--text', action='store_true')
    parser.add_argument('-a', '--audio', action='store_true')
    parser.add_argument('-r', '--recording', action='store_true')
    
    print("Testing the phoneme conversion model")
    
    args = parser.parse_args()
    pcm = PhonemeConversionModel()
    
    if args.text:
        print(args.user_input + ": " + ', '.join(pcm.grapheme_to_phoneme(args.user_input)))
    elif args.audio:
        print("audio file")
        if args.user_input == "This is the default.":
            wav, sr = torchaudio.load('../../data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac')
            #print('wav:{} sr:{}'.format(wav.tolist(), sr))
            transcript, phones = pcm.audio_to_phoneme(wav.tolist(), sr)
            print("transcript: {}, phones: {}".format(transcript, phones))
        else:
            wav, sr = torchaudio.load(args.user_input)
            #print('wav:{} sr:{}'.format(wav.tolist(), sr))
            transcript, phones = pcm.audio_to_phoneme(wav.tolist(), sr)
            print("transcript: {}, phones: {}".format(transcript, phones))

    elif args.recording:
        print("Recording not implemented")

    else:
        print(args.user_input + ": " + ', '.join(pcm.grapheme_to_phoneme(args.user_input)))