import os
from typing import Tuple

import numpy as np
from samples import Samples
from scipy.fftpack import dct
from scipy.signal import stft
from scipy.signal.windows import triang


class Window:
    """Precomputed mel-frequency window."""

    def __init__(
        self,
        mel_range: Tuple[float, float],
        hz_range: Tuple[float, float],
        idx_range: Tuple[int, int],
        weights: np.ndarray,
        segment_max_count: int,
    ) -> None:
        assert len(weights.shape) == 1
        assert weights.shape[0] == idx_range[1] - idx_range[0]

        self.__mel_range = mel_range
        self.__hz_range = hz_range
        self.__idx_range = idx_range

        self.__weights = np.zeros((len(weights), segment_max_count))
        for segment_idx in range(segment_max_count):
            self.__weights[:, segment_idx] = weights

    @property
    def mel_range(self) -> Tuple[float, float]:
        return self.__mel_range

    @property
    def hz_range(self) -> Tuple[float, float]:
        return self.__hz_range

    @property
    def idx_range(self) -> Tuple[int, int]:
        return self.__idx_range

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    def __str__(self) -> str:
        return f"[Window] range: {self.mel_range} mel or {self.hz_range} hz or {self.idx_range} idx; shape: {self.weights.shape}"


class Context:
    def __init__(self, rate: int, segment_ms: float) -> None:
        self.__rate = rate
        self.__segment_ms = segment_ms
        self.__windows = list[Window]()

    @property
    def rate(self) -> int:
        return self.__rate

    @property
    def segment_ms(self) -> float:
        return self.__segment_ms

    @property
    def segment_n(self) -> int:
        return int(self.__segment_ms / 1000 * self.__rate)

    @property
    def segment_max_count(self) -> int:
        assert len(self.__windows) > 0
        return self.__windows[0].weights.shape[1]

    @property
    def mfcc_count(self) -> int:
        return len(self.__windows)

    @property
    def windows(self) -> list[Window]:
        return self.__windows

    def __str__(self) -> str:
        s = (
            f"[Context] rate: {self.__rate} Hz; segment: {self.segment_n} or {self.__segment_ms} ms;"
            f" segment_max_count: {self.segment_max_count} or ≅ {self.segment_max_count / 2 * self.__segment_ms} ms; mfcc_count: {self.mfcc_count}"
        )
        for window in self.__windows:
            s += f"{os.linesep}  {window}"
        return s


def _get_segment_max_count(samples: Samples, context: Context) -> int:
    # TODO-1 Оценка максимального количества сегментов в MFCC (максимальной ширины матрицы MFCC)

    # max_len_sample = max(max(len(data) for data in value) for value in samples.dict.values())

    max_len_sample = 0
    for value in samples.dict.values():
        local_max_len_sample = max(len(data) for data in value)
        if local_max_len_sample > max_len_sample:
            max_len_sample = local_max_len_sample

    max_len_sample *= 1.1
    overlap = context.segment_n // 2
    max_count = int((max_len_sample - context.segment_n) // (context.segment_n - overlap) + 1)
    return max_count


def mel_to_hz(mel: float) -> float:
    # TODO-2 Конвертация мел в герцы
    return 700 * (np.exp(mel/1127) - 1)


def build_context(
    samples: Samples, segment_ms: float, band_width_mel: float, mfcc_n: int
) -> Context:
    context = Context(samples.rate, segment_ms)

    segment_max_count = _get_segment_max_count(samples, context)

    f, _, _ = stft(np.zeros(samples.rate), fs=samples.rate, nperseg=context.segment_n)

    for mfcc_idx in range(mfcc_n):
        mel_range = (
            band_width_mel / 2 * mfcc_idx,
            band_width_mel / 2 * mfcc_idx + band_width_mel,
        )
        hz_range = (mel_to_hz(mel_range[0]), mel_to_hz(mel_range[1]))
        idx_np = np.searchsorted(f, hz_range)
        idx_range = (idx_np[0], idx_np[1])
        weights = triang(idx_range[1] - idx_range[0])
        window = Window(mel_range, hz_range, idx_range, weights, segment_max_count)
        context.windows.append(window)

    return context


def compute_mfcc(
    data: np.ndarray,
    context: Context,
) -> np.ndarray:
    f, t, z = stft(data, fs=context.rate, nperseg=context.segment_n)
    assert isinstance(f, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert isinstance(z, np.ndarray)

    segment_count = z.shape[1]
    assert segment_count <= context.segment_max_count

    mfcc = np.zeros((len(context.windows), segment_count))

    for window_idx, window in enumerate(context.windows):
        stft_values = (
            z[window.idx_range[0]: window.idx_range[1], :]
            * window.weights[:, :segment_count]
        )

        # TODO-3 (Внутри цикла) Вычислить window-idx-ую строку матрицы MFCC по уже взвешенной полосе спектра stft_values
        mfcc[window_idx, :] = np.log(np.sum(np.power(np.abs(stft_values), 2), axis=0) + 1)

    # TODO-4 (Вне цикла) Вычислить спектр от спектра: применить ДКП к каждому столбцу матрицы mfcc
    mfcc = dct(mfcc, type=3, norm='ortho', axis=0)

    return mfcc
