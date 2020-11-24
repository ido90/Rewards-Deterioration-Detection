
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_quantiles(X, Q=100, assume_sorted=False):
    if not assume_sorted:
        X = sorted(X)
    N = len(X)
    try:
        n = len(Q)
        Q = np.array(Q)
    except:
        Q = np.arange(0,1+1e-6,1/Q)
        Q[-1] = 1
        n = len(Q)
    x = [X[int(q*(N-1))] for q in Q]
    return x, Q

def plot_quantiles(x, ax, Q=100, plot_mean=False, **kwargs):
    x, q = get_quantiles(x, Q)
    h = ax.plot(100*q, x, '-', **kwargs)
    if plot_mean:
        ax.axhline(np.mean(x), linewidth=0.8, linestyle='--', color=h[0].get_color())

def labels(ax, xlab=None, ylab=None, title=None, fontsize=12):
    if isinstance(fontsize, int):
        fontsize = 3*[fontsize]
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=fontsize[0])
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=fontsize[1])
    if title is not None:
        ax.set_title(title, fontsize=fontsize[2])

class Axes:
    def __init__(self, N, W=2, axsize=(5,3.5), grid=1, fontsize=13):
        self.fontsize = fontsize
        self.N = N
        self.W = W
        self.H = int(np.ceil(N/W))
        self.axs = plt.subplots(self.H, self.W, figsize=(self.W*axsize[0], self.H*axsize[1]))[1]
        for i in range(self.N):
            if grid == 1:
                self[i].grid(color='k', linestyle=':', linewidth=0.3)
            elif grid ==2:
                self[i].grid()
        for i in range(self.N, self.W*self.H):
            self[i].axis('off')

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if self.H == 1 and self.W == 1:
            return self.axs
        elif self.H == 1 or self.W == 1:
            return self.axs[item]
        return self.axs[item//self.W, item % self.W]

    def labs(self, item, *args, **kwargs):
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.fontsize
        labels(self[item], *args, **kwargs)

def to_device(model, tensor_list):
    if model is not None:
        model.to(DEVICE)
    for i,tns in enumerate(tensor_list):
        if tns is not None:
            if type(tns) is not torch.Tensor:
                tns = torch.from_numpy(tns)
            tensor_list[i] = tns.to(DEVICE)
    return tensor_list

def clean_device(tensor_list):
    for i in reversed(list(range(len(tensor_list)))):
        del tensor_list[i]
    gc.collect()
    torch.cuda.empty_cache()

def update_dict(base_dict, dict_to_add=None, force=False, copy=False):
    if copy:
        base_dict = base_dict.copy()
    if dict_to_add is not None:
        for k, v in dict_to_add.items():
            if force or k not in base_dict:
                base_dict[k] = v
    return base_dict
