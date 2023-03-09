# This is just a temporary file on how to use encodec
from encodec import EncodecModel
from encodec.utils import convert_audio
import torch
import torchaudio

model = EncodecModel.encodec_model_24khz()
path = '../../data/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac'
wav, sr = torchaudio.load(path)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
print(codes)
print(codes.shape)

with torch.no_grad():
    decoded_frames = model.decode(encoded_frames)
decoded_frames = decoded_frames.squeeze(0)
print(decoded_frames.shape)

#---------------------------------------------

from IPython.display import Audio, display
def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

play_audio(decoded_frames, model.sample_rate)