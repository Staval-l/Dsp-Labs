import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.io.wavfile import write


# f - частота, t - время, waveform - форма, fs - частота дискретизации
def tone(f: float, t: int, waveform: str, fs: int = 44100):
    t1 = np.linspace(0, t, fs * t)
    if waveform == 'harmonic':
        x = np.sin((2 * np.pi * f) * t1)
    elif waveform == 'square':
        x = scipy.signal.square((2 * np.pi * f) * t1)
    elif waveform == 'sawtooth_w':
        x = scipy.signal.sawtooth((2 * np.pi * f) * t1, 0.01)
    elif waveform == 'sawtooth':
        x = scipy.signal.sawtooth((2 * np.pi * f) * t1, 1)
    # plt.xlim(0, 0.005)
    # plt.stem(t1, x, use_line_collection=True)
    # plt.show()
    return x


def musical_tone(f: float, t: int, waveform: str, fs: int = 44100, db: float = 0):
    t1 = tone(f, t, waveform, fs)
    for ton in range(2 * int(f), 2000, int(f)):
        t1 += tone(ton, t, waveform, fs)
    t1 /= t1.max()
    a = (10 ** (db / 10)) ** (1 / t * fs)
    for n in range(t1.size):
        t1[n] = (a ** n) * t1[n]
    return t1


f = 430
t = 5
waveform = 'harmonic'
fs = 44100
db = -0.0000001

# y = tone(f, t, waveform, fs)
y = musical_tone(f, t, waveform, fs, db)
write('example.wav', fs, y)
