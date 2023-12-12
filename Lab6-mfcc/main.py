import sys
from datetime import timedelta
from time import time_ns
from typing import Final, Tuple

import librosa
import numpy as np
from mfcc import build_context, compute_mfcc
from samples import load_samples
from scipy.io.wavfile import read

TEMPLATES_DIR: Final[str] = "data"
TEST_FILE: Final[str] = "m.wav"

SEGMENT_LENGTH_MS: Final[float] = 20.0
BAND_WIDTH_MEL: Final[float] = 450.0
MFCC_N = 8

WEIGHT_THRESHOLD = 3.7


def test(test_file: str) -> None:
    # load audio samples

    samples = load_samples(TEMPLATES_DIR)
    print(samples)
    print()

    # build context

    context = build_context(samples, SEGMENT_LENGTH_MS, BAND_WIDTH_MEL, MFCC_N)
    print(context)
    print()

    # compute mfcc for audio samples

    mfcc_list = list[Tuple[str, np.ndarray]]()

    t1 = time_ns()

    for key, data_list in samples.dict.items():
        for data in data_list:
            mfcc_list.append((key, compute_mfcc(data, context)))

    t2 = time_ns()

    print(f"Compute {len(mfcc_list)} mfcc in {(t2-t1) / 1000000} ms")

    # matching

    rate, data = read(test_file)
    assert isinstance(rate, int)
    assert isinstance(data, np.ndarray)

    bulk_n = (context.segment_max_count // 2 - 1) * context.segment_n
    idx = bulk_n

    while idx < len(data):
        segment_time_start = (idx - bulk_n) / rate
        segment_time_end = idx / rate
        print(f"\nВременной промежуток сегмента: {segment_time_start} - {segment_time_end}")

        current_mfcc = compute_mfcc(data[idx - bulk_n : idx], context)

        min_weight = sys.float_info.max
        word = ""
        start_word_index = 0
        for key, pattern_mfcc in mfcc_list:
            # TODO-5 Вычислить вес (меру несхожести) между матрицами current_mfcc (анализируемый фрагмент) и pattern_mfcc (образец произнесения слова key).
            # Из всех итераций цикла необходимо выбрать key/pattern_mfcc с минимальным весом.

            dtw_matrix, wp = librosa.sequence.dtw(pattern_mfcc, current_mfcc, subseq=True, metric="cosine")

            current_width = wp[0, 1] - wp[-1, 1]
            coefficient = current_width / dtw_matrix.shape[0]
            if coefficient < 0.7 or coefficient > 1.3:
                continue

            current_weight = dtw_matrix[wp[0, 0], wp[0, 1]]
            if current_weight < min_weight:
                min_weight = current_weight
                word = key
                start_word_index = wp[-1, 1]

        # TODO-5 Если минимальный вес не превышет некоторый экспериментально подобранный порог,
        # то считать, что в анализируемом фрагменте обнаружен образец.
        # Рассчитать время начала этого образца в глобальном времени (в секундах относительно data) и вывести распозанное слово (key) и временную метку его произношения.
        if min_weight < WEIGHT_THRESHOLD:
            time = (idx - bulk_n + start_word_index) / rate
            print(f"Распознанное слово: {word} Время произношения: {time}")

        idx += bulk_n // 4


if __name__ == "__main__":
    test(TEST_FILE)

    # x = np.array([[5, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 5, 3, 7, 5], [5, 5, 3, 4, 5]])
    # y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # f, t, z = stft(y, nperseg=2, boundary=None, padded=False)
    # print(x)
    # print(x.shape)
    # print(z)
    # print(z.shape)

    # print(dct(x, type=3, norm="ortho", axis=0))

    # print(x.sum(axis=0))
    # y = np.array([[3, 3, 3, 4, 5], [1, 4, 3, 4, 5], [1, 3, 3, 4, 5]])
    # x = np.array([5, 3, 1, 5, 5, 9, 9, 3, 1, 3, 7])
    # y = np.array([2, 6, 4, 5, 8, 2, 2])
    # print(x)
    # print(y)
    #
    # a, b = librosa.sequence.dtw(y, x, subseq=True)
    # print(a.shape)
    # print(a)
    # print(b)
    #
    # a, b = librosa.sequence.dtw(y, x)
    # print(a.shape)
    # print(a)
    # print(b)
    # print(a[b[0,0], b[0, 1]])