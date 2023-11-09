# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-few-public-methods

import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
from scipy.fft import fft
from scipy.io.wavfile import read


def dft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X


def real_stft(x: np.ndarray, segment: int, overlap: int) -> np.ndarray:
    n = x.shape[0]
    assert len(x.shape) == 1
    assert segment < n
    assert overlap < segment

    stft_result = np.array([])
    number_of_segments = (len(x) - segment) // (segment - overlap) + 1
    for i in range(0, number_of_segments):
        slice = x[(segment - overlap) * i:(segment - overlap) * i + segment]
        if len(slice) < segment:
            slice = np.concatenate((slice, np.zeros(segment - len(slice))))
        my_dft = np.resize(dft(slice), (segment // 2 + 1, 1))
        if len(stft_result) == 0:
            stft_result = my_dft
        else:
            stft_result = np.concatenate((stft_result, my_dft), axis=1)
    return stft_result


class Test(unittest.TestCase):
    class Params:
        def __init__(self, n: int, segment: int, overlap: int) -> None:
            self.n = n
            self.segment = segment
            self.overlap = overlap

        def __str__(self) -> str:
            return f"n={self.n} segment={self.segment} overlap={self.overlap}"

    def test_dft(self) -> None:
        for n in (10, 11, 12, 13, 14, 15, 16):
            with self.subTest(n=n):
                np.random.seed(0)
                x = np.random.rand(n) + 1j * np.random.rand(n)
                actual = dft(x)
                expected = fft(x)
                self.assertTrue(np.allclose(actual, expected))

    #@unittest.skip
    def test_stft(self) -> None:
        params_list = (
            Test.Params(50, 10, 5),
            Test.Params(50, 10, 6),
            Test.Params(50, 10, 7),
            Test.Params(50, 10, 8),
            Test.Params(50, 10, 9),
            Test.Params(101, 15, 7),
            Test.Params(101, 15, 8),
        )

        for params in params_list:
            with self.subTest(params=str(params)):
                np.random.seed(0)
                x = np.random.rand(params.n)
                actual = real_stft(x, params.segment, params.overlap)
                _, _, expected = stft(
                    x,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    padded=False,
                    window="boxcar",
                )
                assert isinstance(expected, np.ndarray)
                self.assertTrue(np.allclose(actual, params.segment * expected))


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    fs, data = read("2023_lab4_2.wav")
    data = [r[0] for r in data]
    f, t, spectrum = stft(data, fs, nperseg=4000)
    plt.figure('Spectrogram')
    plt.pcolormesh(t, f, np.abs(spectrum) ** 2)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.show()
    main()
