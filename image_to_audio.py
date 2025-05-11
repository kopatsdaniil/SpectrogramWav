import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import zoom


def inverse_spectrogram(
    spec_npy='output/spectrogram.npy',
    output_audio='reconstructed_wav/reconstructed.wav',
    speed=2.0,
    pitch_shift=12.0,
    sr=22050
):
    stft = np.load(spec_npy)

    time_axis_scale = 1 / speed
    stft_scaled = zoom(stft, (1, time_axis_scale), order=3)

    y = librosa.istft(stft_scaled)

    if pitch_shift != 0.0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

    sf.write(output_audio, y, samplerate=sr)
    print(f"Сохранено: {output_audio}")


if __name__ == "__main__":
    inverse_spectrogram()
