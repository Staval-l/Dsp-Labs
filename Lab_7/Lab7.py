from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, deconvolve
from scipy.fft import fft
from statsmodels.tsa.stattools import acf

MORSE_CODES: Final[dict[str, str]] = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
}


N = 51


def morse_encode(message: str, unit_length: int) -> np.ndarray:
    dot = np.ones(unit_length)
    dash = np.ones(3 * unit_length)
    letter_intraspace = np.zeros(unit_length)
    letter_interspace = np.zeros(3 * unit_length)
    word_interspace = np.zeros(7 * unit_length)

    encoded = np.zeros((0))
    prev_symbol_is_letter = False

    for symbol in message:
        if symbol in MORSE_CODES:
            if prev_symbol_is_letter:
                encoded = np.concatenate((encoded, letter_interspace))
            is_first = True
            for code in MORSE_CODES[symbol]:
                if is_first:
                    is_first = False
                else:
                    encoded = np.concatenate((encoded, letter_intraspace))
                if code == ".":
                    chunk = dot
                elif code == "-":
                    chunk = dash
                else:
                    msg = f'Invalid morse code "{MORSE_CODES[symbol]}" for "{symbol}".'
                    raise Exception(msg)
                encoded = np.concatenate((encoded, chunk))
            prev_symbol_is_letter = True
        else:
            encoded = np.concatenate((encoded, word_interspace))
            prev_symbol_is_letter = False

    return encoded


def build_lowpass(w: float, n: int) -> np.ndarray:
    # TODO 2.3 Рассчитать ИХ для ФР НЧ КИХ-фильтра с частотой среза w [рад/отсчёт] и размером n [отсчётов]
    ih_before_zero = np.sin(w*np.arange(-n//2, 0)) / (np.arange(-n//2, 0) * np.pi)
    ih_after_zero = np.sin(w*np.arange(1, n//2 + 1)) / (np.arange(1, n//2 + 1) * np.pi)
    filter_ih = np.concatenate([ih_before_zero, [w/np.pi], ih_after_zero])

    plt.plot(filter_ih)
    plt.title("ФР НЧ КИХ-фильтр")
    plt.show()
    return filter_ih


class LowpassReconstruction:
    def __init__(self, recovered: np.ndarray, unit_len: int) -> None:
        self.__recovered = recovered
        self.__unit_len = unit_len

    @property
    def recovered(self) -> np.ndarray:
        return self.__recovered

    @property
    def unit_len(self) -> int:
        return self.__unit_len


def lowpass_reconstruct(y: np.ndarray, h: np.ndarray) -> LowpassReconstruction:
    # TODO 2.1 Развернуть y и h, чтобы получить оценку x
    x_1, rem = deconvolve(y, h)
    # print(f"Остаток: {np.max(np.abs(rem))}")

    # TODO 2.2 Определить размер одной точки Морзе в отсчётах и соответствующую частоту среза w
    spectrum = fft(x_1 - np.mean(x_1))
    plt.plot(np.abs(spectrum))
    plt.title("Спектр зашумленного полезного сигнала")
    plt.show()

    index_max = np.argmax(np.abs(spectrum[:int(spectrum.shape[0] / 2)]))
    print(f"Index max: {index_max}")
    w0 = 2 * np.pi * index_max / x_1.shape[0]
    print(f"wo = {w0}")
    M = int(np.round(x_1.shape[0] / (2 * index_max)))
    print(f"Размер одной точки: {M}")

    # TODO 2.4 Построить и применить ФР НЧ КИХ-фильтр

    h_r = build_lowpass(w0, N)
    x_r = convolve(x_1, h_r)

    plt.plot(x_r)
    plt.title("Восстановленный сигнал (ФР НЧ КИХ-фильтр)")
    plt.show()

    del y, h
    return LowpassReconstruction(x_r, M)


class SuboptimalReconstruction:
    def __init__(self, recovered: np.ndarray) -> None:
        self.__recovered = recovered
        # Добавить допданные при необходимости

    @property
    def recovered(self) -> np.ndarray:
        return self.__recovered


def suboptimal_reconstruct(
    y: np.ndarray, h: np.ndarray, v: np.ndarray
) -> SuboptimalReconstruction:
    # TODO 4.1 Оценить r_y, r_v
    r_y = np.var(y) * acf(y, nlags=100)
    r_y = np.concatenate((np.flip(r_y), r_y[1:]))
    plt.plot(r_y)
    plt.title("R_y")
    plt.show()

    r_v = np.var(v) * acf(v, nlags=100)
    r_v = np.concatenate((np.flip(r_v), r_v[1:]))
    plt.plot(r_v)
    plt.title("R_v")
    plt.show()

    # TODO 4.2 Оценить r_xy, r_x
    r_xy, _ = deconvolve(np.flip(r_y-r_v), h)
    r_xy = np.flip(r_xy)
    plt.plot(r_xy)
    plt.title("R_xy")
    plt.show()


    r_x, _ = deconvolve(r_xy, h)
    plt.plot(r_x)
    plt.title("R_x")
    plt.show()

    # print(f"len r_xy: {r_xy.shape[0]} \n"
    #       f"len r_x: {r_x.shape[0]} \n"
    #       f"argmax r_x: {r_x.argmax()} \n"
    #       f"len h: {h.shape[0]} \n"
    #       f"len r_y: {r_y.shape[0]}")

    # TODO 4.3 Рассчитать (уравнение Винера-Хопфа) фильтр и применить фильтр

    # A*x = b
    # Искажающая ИХ не сдвинута
    # Если R_x не был сдвинут, то R_xy был бы не сдвинут
    # Так как R_x сдвинут, то при свертке получаем, что R_xy будет сдвинут на ту же задержку
    center_r_xy = r_x.shape[0] // 2
    center_r_y = r_y.shape[0] // 2
    D = np.arange(-N // 2, N // 2)

    A = np.zeros((N, N))
    for i, m in enumerate(D):
        for j, k in enumerate(D):
            A[i, j] = r_y[center_r_y + k - m]

    b = np.flip(r_xy[center_r_xy - N//2: center_r_xy + N//2 + 1])

    # print(f"A shape is {A.shape}")
    # print(f"b shape is {b.shape}")

    h_rec = np.linalg.solve(A, b)

    plt.plot(h_rec)
    plt.title("Винера хопфа")
    plt.show()

    x_rec = convolve(y, h_rec)

    # TODO 4.4 По r_x[0] и ИХ фильтра рассчитать оценку погрешности восстановления

    # так как задержка r_xy = задержка r_x
    error = r_x[center_r_xy] - np.dot(h_rec, b)
    print(f"Средняя ошибка восстановления: {error}")
    del y, v, h
    return SuboptimalReconstruction(x_rec)


def main() -> None:
    # TODO 1 Загрузить данные и оценить h[n]
    data = np.load("24.npy")
    y = np.ravel(data[0, :])
    v = np.ravel(data[1, :])
    h = data[2:, :]

    h = h.mean(axis=0)

    x = np.arange(0, len(h[0:200]))
    plt.plot(x, h[0:200])
    plt.title("ИХ искажающей системы")
    plt.show()

    h[np.abs(h) < 0.1] = 0
    h = np.trim_zeros(h, "b")
    print(f"h shape: {h.shape[0]}")

    # TODO 3.1 Восстановить сигнал с помощью lowpass_reconstruct и вручную/автоматически декодировать сообщение
    rec_signal_lowpass = lowpass_reconstruct(y, h)
    M = rec_signal_lowpass.unit_len

    processed_signal = np.copy(rec_signal_lowpass.recovered)
    processed_signal[processed_signal < 0.5] = 0
    processed_signal[processed_signal >= 0.5] = 1
    plt.plot(processed_signal)
    plt.title(f"После пороговой обработки")
    plt.show()

    # TODO 3.2 С помощью morse_encode сформировать идеальный полезный сигнал и рассчитать MSE
    message = "a bad workman always blames his tools" # свою фразу
    encoded_message = morse_encode(message, M)

    # print(f"Length of encoded_message: {encoded_message.shape[0]}")
    # print(f"Length of recovered signal: {rec_signal_lowpass.recovered.shape[0]}")

    shift = N // 2  # Задержка вносимая фильтром
    n = encoded_message.shape[0]
    mse = np.sum(np.power((encoded_message - rec_signal_lowpass.recovered[shift:shift + n]), 2)) / n
    print(f"MSE (НЧ КИХ-фильтра): {mse}")

    # TODO 5 Восстановить сигнал с помощью suboptimal_reconstruct

    rec_signal_suboptimal = suboptimal_reconstruct(y, h, v)
    plt.plot(rec_signal_suboptimal.recovered)
    plt.title("Восстановленный сигнал (Винера)")
    plt.show()

    mse = np.sum(np.power((encoded_message - rec_signal_suboptimal.recovered[shift:shift + n]), 2)) / n
    print(f"MSE (Квазиоптимальное восстановление с помощью фильтра Винера): {mse}")


if __name__ == "__main__":
    main()
