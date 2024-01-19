import numpy as np
import torch as th


DTYPE = th.float32


def to_tensor(arr, dtype=DTYPE):
    return th.as_tensor(arr, dtype=dtype)


class TensorBuffer:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_dims(cls, **kwargs):
        return cls(**{
            k: th.zeros(v, dtype=DTYPE) for k, v in kwargs.items()
        })
    
    def flatten(self):
        return TensorBuffer(**{
            k: v.flatten(start_dim=0, end_dim=1) for k, v in self.__dict__.items()
        })

    def get(self, *args):
        return TensorBuffer(**{
            k: self[k] for k in args
        })

    def to(self, device):
        for key, value in self.__dict__.items():
            self[key] = value.to(device)

    def cpu(self):
        for key, value in self.__dict__.items():
            self[key] = value.cpu()

    def clone(self):
        return TensorBuffer(**{
            k: v.clone() for k, v in self.__dict__.items()
        })

    def reset(self):
        for key, value in self.__dict__.items():
            self[key] = th.zeros_like(value)

    @property
    def shape(self):
        return next(iter(self.__dict__.values())).shape

    def __len__(self):
        return len(next(iter(self.__dict__.values())))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.__dict__.get(idx)
        else:
            return TensorBuffer(**{
                k: v[idx] for k, v in self.__dict__.items()
            })
    
    def __setitem__(self, idx, val):
        if isinstance(idx, str):
            self.__dict__[idx] = val
        else:
            for i, value in enumerate(self.__dict__.values()):
                value[idx] = val[i]

    def __iter__(self):
        return TensorBufferIterator(self)

    def __repr__(self):
        kv_strs = [f"\n{k:>12}: Tensor{tuple(v.shape)}" for k, v in self.__dict__.items()]
        return "TensorBuffer({}\n)".format("".join(kv_strs))
    

class TensorBufferIterator:

    def __init__(self, buffer):
        self.index = 0
        self.buffer = buffer

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            result = self.buffer[self.index]
        except:
            raise StopIteration
        self.index += 1
        return result
    

def batch_sample(data, batch_size):
    N, M = data.shape[0], data.shape[1]
    random_inds = np.random.permutation(N)

    for batch_start in range(0, N, batch_size // M):
        batch_end = batch_start + batch_size
        batch_inds = random_inds[batch_start:batch_end]
        yield data[batch_inds]


def stack_states(states, next_states):
    pairs = th.stack((states, next_states), dim=states.ndim-1)
    pairs = pairs.flatten(start_dim=pairs.ndim-2)
    return pairs