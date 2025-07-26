import modal
import torch.nn as nn
import torchaudio.transforms as T
import torch

app = modal.App("audio-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-model")

class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )
        
    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()

        waveform = waveform.unsqueeze(0)

        spectrogram = self.transform(waveform)

        return spectrogram.unsqueeze()