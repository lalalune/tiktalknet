import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.tts.helpers.helpers import regulate_loudness
from scipy.io.wavfile import write
import soundfile as sf
from pydub import AudioSegment

# Load the FastPitch model
fastpitch = FastPitchModel.from_pretrained(model_name="FastPitch")

# Load the HiFi-GAN model
hifigan = HifiGanModel.from_pretrained(model_name="HiFiGan")

# The text you want to convert to speech
text = "Hello, this is a test of the text to speech system."

# Convert the text to a spectrogram
with torch.no_grad():
    spectrogram = fastpitch.convert_text_to_spectrogram(text)

# Convert the spectrogram to audio
with torch.no_grad():
    audio = hifigan.convert_spectrogram_to_audio(spectrogram)

# Regulate the loudness of the audio
audio = regulate_loudness(audio)

# Save the audio to a WAV file
sf.write('output.wav', audio.cpu().numpy(), 22050)

# Convert WAV to MP3
sound = AudioSegment.from_wav("output.wav")
sound.export("output.mp3", format="mp3")