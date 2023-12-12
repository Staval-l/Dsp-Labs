import os
from typing import Optional

import numpy as np
from scipy.io.wavfile import read


class Samples:
    def __init__(self, rate: int) -> None:
        self.__dict = dict[str, list[np.ndarray]]()
        self.__rate = rate

    @property
    def dict(self) -> dict[str, list[np.ndarray]]:
        return self.__dict

    @property
    def rate(self) -> int:
        return self.__rate

    def __str__(self) -> str:
        s = f"[Samples] rate: {self.__rate} Hz"
        for key, data_list in self.__dict.items():
            s += f'{os.linesep}  {len(data_list)} items for "{key}": '
            s += ", ".join((str(data.shape[0]) for data in data_list))
        return s


def load_samples(samples_dir: str) -> Samples:
    files = [
        os.path.join(samples_dir, f)
        for f in os.listdir(samples_dir)
        if f.endswith(".wav")
    ]

    samples: Optional[Samples] = None

    for file in files:
        key = os.path.splitext(os.path.basename(file))[0]

        last_hyphen_idx = key.rfind("-")
        if last_hyphen_idx != -1:
            try:
                _ = int(key[last_hyphen_idx + 1:])
                key = key[:last_hyphen_idx]
            except:
                pass

        rate, data = read(file)
        assert isinstance(rate, int)
        assert isinstance(data, np.ndarray)
        if len(data.shape) == 2:
            data = data[:, 0]

        if samples is None:
            samples = Samples(rate)
        elif samples.rate != rate:
            raise Exception("All files must have the same sample rate.")

        if key in samples.dict:
            data_list = samples.dict[key]
        else:
            data_list = list[np.ndarray]()
            samples.dict[key] = data_list

        data_list.append(data)

    if samples is None:
        raise Exception(
            f'No wav file found in directory "{os.path.abspath(samples_dir)}".'
        )

    return samples
