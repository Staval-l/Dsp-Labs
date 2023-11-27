# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-few-public-methods

import unittest

import numpy as np
from scipy.signal import stft, istft
from scipy.fft import ifft
import scipy.io.wavfile as wavfile


def idft(x: np.ndarray) -> np.ndarray:
    # TODO #1: Implement complex-data inverse DFT
    length = x.shape[0]
    result = np.empty(length, dtype=complex)

    for n in range(length):
        exp_array = np.array(np.exp(1j * 2 * np.pi / length * n * np.arange(length)), dtype=complex)
        result[n] = np.dot(x, exp_array) / length

    return result


def real_istft(spectrum: np.ndarray, segment: int, overlap: int) -> np.ndarray:
    assert len(spectrum.shape) == 2
    assert spectrum.shape[0] == segment // 2 + 1

    # TODO #2: Implement inverse STFT

    signal_length = (spectrum.shape[1] - 1) * (segment - overlap) + segment

    result = np.zeros(signal_length)
    idft_segment_matrix = np.zeros((spectrum.shape[1], signal_length))

    for t in range(spectrum.shape[1]):
        dft_segment = spectrum[:, t]

        complex_data_segment = np.array(
            [dft_segment[i] if i < (segment // 2 + 1) else np.conj(dft_segment[segment - i]) for i in range(segment)]
        )
        inverse_dft_segment = idft(complex_data_segment)

        idft_segment_matrix[t, t*(segment - overlap):t*(segment - overlap) + segment] = inverse_dft_segment.real

    for n in range(signal_length):
        sum_value = sum(idft_segment_matrix[:, n])
        count_overlap_segment = 0
        for t in range(spectrum.shape[1]):
            if t*(segment - overlap) <= n <= t*(segment - overlap) + segment - 1:
                count_overlap_segment += 1

        value = sum_value / count_overlap_segment
        result[n] = value

    return result


# class Test(unittest.TestCase):
#     class Params:
#         def __init__(self, n: int, segment: int, overlap: int) -> None:
#             self.n = n
#             self.segment = segment
#             self.overlap = overlap
#
#         def __str__(self) -> str:
#             return f"n={self.n} segment={self.segment} overlap={self.overlap}"
#
#     def test_idft(self) -> None:
#         for n in (10, 11, 12, 13, 14, 15, 16):
#             with self.subTest(n=n):
#                 np.random.seed(0)
#                 x = np.random.rand(n) + 1j * np.random.rand(n)
#                 actual = idft(x)
#                 expected = ifft(x)
#                 self.assertTrue(np.allclose(actual, expected))
#
#     # @unittest.skip
#     def test_istft_unmodified(self) -> None:
#         self._test_istft(False)
#
#     # @unittest.skip
#     def test_istft_modified(self) -> None:
#         self._test_istft(True)
#
#     def _test_istft(self, modify: bool) -> None:
#         params_list = (
#             Test.Params(50, 10, 5),
#             Test.Params(50, 10, 6),
#             Test.Params(50, 10, 7),
#             Test.Params(50, 10, 8),
#             Test.Params(50, 10, 9),
#             Test.Params(101, 15, 7),
#             Test.Params(101, 15, 8),
#         )
#
#         for params in params_list:
#             with self.subTest(params=str(params)):
#                 np.random.seed(0)
#
#                 x = np.random.rand(params.n)
#
#                 _, _, s = stft(
#                     x,
#                     boundary=None,
#                     nperseg=params.segment,
#                     noverlap=params.overlap,
#                     padded=False,
#                     window="boxcar",
#                 )
#
#                 assert isinstance(s, np.ndarray)
#
#                 if modify:
#                     low_pass_filter = np.concatenate(
#                         (
#                             np.ones(s.shape[0] // 2),
#                             np.zeros(s.shape[0] - s.shape[0] // 2),
#                         )
#                     )
#                     for column in np.arange(s.shape[1]):
#                         s[:, column] = s[:, column] * low_pass_filter
#
#                 _, expected = istft(
#                     s,
#                     boundary=None,
#                     nperseg=params.segment,
#                     noverlap=params.overlap,
#                     window="boxcar",
#                 )
#
#                 assert isinstance(expected, np.ndarray)
#
#                 actual = real_istft(s * params.segment, params.segment, params.overlap)
#
#                 self.assertTrue(np.allclose(actual, expected))


def main() -> None:
    # unittest.main()

    # TODO #3: Implement robotic effect using scipy's stft/istft

    fs, x = wavfile.read("L2.wav")

    segment = int(20e-3 * fs)
    overlap = int(7e-3 * fs)
    window = "triangle"

    _, _, stft_voice = stft(x,
                            fs=fs,
                            nperseg=segment,
                            noverlap=overlap,
                            window=window)

    stft_voice = abs(stft_voice)

    _, robotic_voice = istft(stft_voice,
                             fs=fs,
                             nperseg=segment,
                             noverlap=overlap,
                             window=window)

    robotic_voice = robotic_voice / max(abs(max(robotic_voice)), abs(min(robotic_voice)))

    wavfile.write("robotic_voice.wav",
                  rate=fs,
                  data=robotic_voice)


if __name__ == "__main__":
    main()
