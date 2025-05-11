import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def generate_spectrogram(audio_path, spec_npy='output/spectrogram.npy', duration_file='output/duration.txt',
                         spec_image='output/spectrogram.png'):
    y, sr = librosa.load(audio_path, sr=None)

    np.savetxt(duration_file, [len(y)])

    stft = librosa.stft(y)
    np.save(spec_npy, stft)

    plt.figure(figsize=(10, 4))
    d_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    librosa.display.specshow(d_db, sr=sr, x_axis='time', y_axis='log', cmap='gray')

    plt.axis('off')

    plt.tight_layout()
    plt.savefig(spec_image, bbox_inches='tight', pad_inches=0)
    print(f"Сохранены файлы: {spec_npy}, {duration_file}, {spec_image}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        generate_spectrogram(sys.argv[1])
    generate_spectrogram("input/input.mp3")
