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
    #plt.xlim(0, 0.005)
    #plt.stem(t1, x, use_line_collection=True)
    #plt.show()
    return x


def musical_tone(f: float, t: int, waveform: str, fs: int = 44100, db: float = 0):
    t1 = tone(f, t, waveform, fs)
    for ton in range(2 * int(f), 2000, int(f)):
        t1 += tone(ton, t, waveform, fs)
    t1 /= t1.max()
    a = (10 ** (db / 10)) ** (1 / (t * fs))
    for n in range(t1.size):
        t1[n] = (a ** n) * t1[n]
    return t1


def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 440  # Frequency of Note A4
    keys = np.array([x + str(y) for y in range(0, 9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end + 1]

    note_freqs = dict(zip(keys, [2 ** ((n + 1 - 49) / 12) * base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0  # stop
    return note_freqs


def generate_song():
    note_list = []

    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['G4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['D4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['C4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['B3'], t=t, fs=fs, waveform=waveform))

    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['G4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['D4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['C4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['B3'], t=t, fs=fs, waveform=waveform))

    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['G4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['E4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['D4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['C4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['D4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['C4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['B3'], t=t, fs=fs, waveform=waveform))

    note_list.append(tone(f=note_freqs['G4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['G4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['G4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['G4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['A4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['A4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['A4'], t=t, fs=fs, waveform=waveform))
    note_list.append(tone(f=note_freqs['A4'], t=t, fs=fs, waveform=waveform))

    return note_list


def song():
    list = generate_song()
    music = np.zeros(1)
    music = np.concatenate((music, list), axis=None)

    return music


f = 430
t = 1
waveform = 'harmonic'
fs = 16000
db = 0

note_freqs = get_piano_notes()
music = song()
write('example.wav', fs, music)
#y = tone(f, t, waveform, fs)
#y = musical_tone(f, t, waveform, fs, db)
#write('example.wav', fs, y)