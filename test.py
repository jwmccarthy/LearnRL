import attrs
import numpy as np
from typing import List


np_field = attrs.field(converter=np.asarray)


@attrs.define()
class Sample:
    a: np.ndarray
    b: np.ndarray

@attrs.define()
class Data:
    KEYS = ("a", "b")

    a: np.ndarray = np_field
    b: np.ndarray = np_field
    c: np.ndarray = attrs.field(init=False)

    def __getitem__(self, idx):
        return Sample(*[getattr(self, key)[idx] for key in self.KEYS])
    
    def __setitem__(self, idx, val):
        for i, key in enumerate(self.KEYS):
            getattr(self, key)[idx] = val[i]


if __name__ == "__main__":
    d = Data(
        [1, 2, 3],
        [4, 5, 6]
    )
    print(d)
    d.a = [7, 8, 9]
    d[0] = [1, 2]
    print(getattr(d, "a"))
    print(d)
    print(d[1:3])