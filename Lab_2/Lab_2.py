import matplotlib.pyplot as plt
import numpy as np
import pyreaper
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write
from scipy.signal.windows import triang
from statsmodels.tsa.stattools import acf


def my_acf(x, m):
    n = x.shape[0]
    summ = 0
    sample_mean = np.sum(x) / n
    for k in range(0, n-m):
        summ += (x[k] - sample_mean) * (x[k+m] - sample_mean)
    return summ / ((n-m) * np.var(x))


def create_exp(fs, f, length):
    return np.array(np.exp(-1j*2*np.pi * f / fs * np.arange(length)), dtype="complex_")


def my_dtft(x, fs, f):
    if hasattr(f, '__iter__'):
        result = np.empty(f.shape[0])
        for i, value in enumerate(f):
            result[i] = np.absolute(np.dot(x, create_exp(fs, value, x.shape[0])))
        return result
    else:
        result = np.absolute(np.dot(x, create_exp(fs, f, x.shape[0])))
        # вывести 200 примерно
        return result


def psola(x, fs, k):
    int16_info = np.iinfo(np.int16)
    x = x * min(int16_info.min, int16_info.max)
    x = x.astype(np.int16)

    pm_times, pm, f_times, f, _ = pyreaper.reaper(x, fs)

    T = round(fs / np.mean(f[f != -1]))
    n = x.shape[0] // T

    result = np.zeros(round(T*(k*(n-1)+2)))

    window_func = triang(2*T)

    for step in range(0, x.shape[0] - 2*T, T):
        src_start = round(step)
        dst_start = round(step * k)
        result[dst_start: dst_start + 2*T] += x[src_start: src_start + 2*T] * window_func

    max_value = max(abs(max(result)), abs(min(result)))
    result /= max_value
    return result


def draw_graph(x, y):
    plt.plot(x, y)
    plt.show()


def read_file(filename):
    fs, x = wavfile.read(filename)
    # x = np.array(x)
    # x = [r[0] for r in x]
    # x = np.array(x)
    x = x.astype(float) / max(abs(min(x)), abs(max(x)))
    t = np.linspace(0, (len(x) - 1) / fs, len(x))
    return x, t, fs


def task_1(x, show=False):
    res_acf = acf(x, adjusted=True, nlags=x.shape[0])
    res_my_acf = np.array([my_acf(x, m) for m in range(0, x.shape[0])])

    if show is True:
        print(f'Результат функции acf:\n{res_acf}:')
        print(f'Результат функции my_acf:\n{res_my_acf}')

    m_space = np.linspace(0, res_my_acf.shape[0] - 1, res_my_acf.shape[0])
    draw_graph(m_space, res_my_acf)

    for i in range(1, len(res_acf)-1):
        if res_acf[i - 1] < res_acf[i] > res_acf[i + 1]:
            print(f'm_max: {i}')


def task_2(x, fs):
    sp = my_dtft(x, fs, np.arange(40, 500, 1))
    w_space = np.arange(40, 500)
    draw_graph(w_space, sp)


def task_3(x, t, fs):
    # Подготовка данных для reaper
    int16_info = np.iinfo(np.int16)
    x = x * min(int16_info.min, int16_info.max)
    x = x.astype(np.int16)
    # Вызов reaper
    pm_times, pm, f_times, f, _ = pyreaper.reaper(x, fs)
    # Отображение позиций пиков
    plt.figure('[Reaper] Pitch Marks')
    plt.plot(t, x)
    plt.scatter(pm_times[pm == 1], x[(pm_times * fs).astype(int)][pm == 1], marker='x', color='red')
    # Отображение значений основной частоты
    plt.figure('[Reaper] Fundamental Frequency')
    plt.plot(f_times, f)
    print('Average fundamental frequency:', np.mean(f[f != -1]))
    plt.show()


def task_4(x, fs, k):
    res = psola(x, fs, k)
    write("res_voice.wav", fs, res)


def task_5():
    x, t, fs = read_file("example.wav")
    t_space = np.arange(680, 720, 0.1)
    y = my_dtft(x, fs, t_space)
    # t_space = t_space * 2 * np.pi / fs
    draw_graph(t_space, y)


def main():
    x, t, fs = read_file("L2.wav")
    slice = x[fs:fs+5000]

    task_1(slice, True)
    task_2(x, fs)
    task_3(x, t, fs)
    task_4(x, fs, 1.5)
    #task_5()


if __name__ == '__main__':
    main()
