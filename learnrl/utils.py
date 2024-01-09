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
            k: self.__dict__.get(k) for k in args
        })

    def gpu(self, device="cuda"):
        for value in self.__dict__.values():
            value = value.to(device)

    def cpu(self):
        for value in self.__dict__.values():
            value = value.cpu()

    def reset(self):
        for value in self.__dict__.values():
            value = th.zeros_like(value)

    def __getitem__(self, idx):
        return TensorBuffer(**{
            k: v[idx] for k, v in self.__dict__.items()
        })
    
    def __setitem__(self, idx, val):
        for i, value in enumerate(self.__dict__.values()):
            value[idx] = val[i]

    def __repr__(self):
        kv_strs = [f"\n    {k}: {v.shape}" for k, v in self.__dict__.items()]
        return "TensorBuffer({}\n)".format(",".join(kv_strs))