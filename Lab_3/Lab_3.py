import numpy as np

from scipy.interpolate import interp1d
from scipy.io.wavfile import read, write

def shift(x, fs, dt, at, f):
    t = np.linspace(0, (len(x) - 1) / fs, len(x))
    new_t = np.zeros(len(t))
    for i in range(len(new_t)):
        new_t[i] = t[i] + dt + at * np.sin(2 * np.pi * f * t[i])
    f = interp1d(new_t, x)
    result = np.zeros(int(len(new_t)))
    min_t = min(new_t)
    for i in range(len(new_t)):
        if t[i] >= min_t:
            result[i] = f(t[i])
    return result



fs, data = read("Lab_2.wav")

data = [r[0] for r in data]
data = np.array(data)
data = data.astype(np.float32) / max(abs(min(data)), abs(max(data)))

data1 = data.astype(np.float32) / data.max()
data1 += shift(data, fs, 0.01, 0.02, 0.5)
data2 = data.astype(np.float32) / data.max()
data2 += shift(data, fs, 0.04, 0.02, 0.5)

data1 /= data1.max()
result = data + data1 + data2
result /= result.max()
write("result_horus.wav", fs, result)
