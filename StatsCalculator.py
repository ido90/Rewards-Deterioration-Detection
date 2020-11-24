
'''
Contents:
GENERAL FUNCTIONS
PARENT CLASS
    - implements t-test & bootstrap test wrt abstract test-statistic
    - initialized with reference data, from which come the estimates of means, variances & bootstrap distributions
    - supports data downsampling (though not too efficiently... if possible, downsampling should be done in advance)
CHILDREN CLASSES
    - MixTest (MDT): uses min(p-val) over the p-values of multiple statistics (any of the statistics below can be used)
    - iidMean: statistic=mean, variance assumes iid data
    - IndependentMean: statistic=mean, variance assumes identically-distributed data (but not independent)
    - SimpleMean: statistic=mean, variance is based on Cov-matrix, assuming neither independence nor identicality
    - IndependentWeightedMean: statistic=mean/var or mean/std
    - CUSUM
    - Hotelling: statistic = (x-mu)*S*(x-mu), where S is the inverse-covariance matrix
    - WeightedMean (UDT): statistic=mean(S*x), where S is the inverse-covariance matrix
    - CVaR: statistic=mean of p% worst elements of x (centralized compared to the reference means)
    - TransformedCVaR (PDT): mean of p% worst elements of x after inverse-Sigma transformation

Ido Greenberg, 2020
'''

import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from pathlib import Path
import pickle as pkl
from warnings import warn
import gc
import torch
import utils

BOOTSTRAP_PATH = Path('Spectation/bootstrap')

################   GENERAL FUNCTIONS   ################

def GramSchmidtVars(Sigma, eps=1e-10):
    T = len(Sigma)
    G = np.zeros_like(Sigma)
    for i in range(T):
        for j in range(i):
            # Sigma[i,j] = <G[i,:j+1],G[j,:j+1]> = sum(G[i,:j+1]*G[j,:j+1]) = sum(G[i,:j-1]*G[j,:j-1]) + Gij*Gjj
            G[i, j] = (Sigma[i, j] - np.sum(G[j, :j] * G[i, :j])) / G[j, j] if G[j, j] != 0 else 0
        tmp = Sigma[i, i] - np.sum(G[i, :i] ** 2)
        if tmp < -eps:
            raise ValueError()
        tmp = max(tmp, 0)
        G[i, i] = np.sqrt(tmp)
    return G

def cusum(X, M=0, S=1, K=0.5, do_plot=False, ax=None):
    if type(S) not in (list, tuple, np.ndarray):
        S = len(X) * [S]
    if type(M) not in (list, tuple, np.ndarray):
        M = len(X) * [M]

    cm = np.zeros(len(X))
    cp = np.zeros(len(X))

    cm[0] = np.min((X[0] - M[0]) / S[0])
    cp[0] = np.max((X[0] - M[0]) / S[0])
    for i, (x, m, s) in enumerate(zip(X[1:], M[1:], S[1:])):
        cm[i + 1] = min(0, cm[i] + (x - m) / s + K)
        cp[i + 1] = max(0, cp[i] + (x - m) / s - K)

    if do_plot:
        if ax is None:
            ax = utils.Axes(1, 1)[0]
        ax.plot(X-M, label='x-m')
        ax.plot(cm, label='Cm')
        ax.plot(cp, label='Cp')
        utils.labels(ax, 't')
        ax.legend()

    return cm, cp

################   PARENT CLASS   ################
# t-test & bootstrap test wrt abstract test-statistic

class StatisticCalculator:
    def __init__(self, X, resolution=1, T=None, B=2000, seed=None, side=-1, force_side=False,
                 use_torch=False, gpu=False, max_ref_data=None, title='', verbose=0):
        self.seed = seed
        self.side = side
        self.force_side = force_side
        self.torch = use_torch
        self.gpu = gpu
        self.resolution = resolution
        if verbose>=1 and self.resolution>1:
            warn('resolution>1 is currently highly inefficient, since all intervals of data are averaged for every calculation. ' + \
                 'better to average the samples in all data in advance and then run this with resolution=1.')
        if max_ref_data is not None:
            X = X[:max_ref_data, :]
        X = self.downsample(X)
        self.X0 = X
        self.T = T
        self.B = B
        self.boot = {}
        self.boot2 = {}
        self.title = title
        self.rolling_statistic = {}

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        if seed is not None:
            np.random.seed(seed)

    def clean_gpu(self, level=0):
        torch.cuda.empty_cache()

    def calc(self, X, side=None, bs=False):
        raise NotImplementedError()
    def calcWrap(self, X, ds_fun=None, **kwargs):
        return self.calc(self.downsample(X, ds_fun), **kwargs)
    def calc2(self, X1, X2, side=None, bs=False):
        return self.calc(X2-X1,side=side,bs=bs)
    def calcMean(self, n):
        raise NotImplementedError()
    def calcVar(self, n):
        raise NotImplementedError()
    # def rolling_calc(self, X, key):
    #     raise NotImplementedError()
    def get_temporal_weights(self):
        raise NotImplementedError()

    def rolling_calc(self, X, key, new_samples=1, **kwargs):
        return self.calc(X, **kwargs)

    def reset_rolling_statistic(self, keys=None):
        if keys is None:
            self.rolling_statistic = {}
        else:
            for k in keys:
                self.rolling_statistic[k] = None

    def save_bootstrap(self, fname=None):
        if fname is None:
            fname = f'bootstraps_{len(self.boot):d}_{self.title:s}_res{self.resolution:d}_T{self.T:d}_B{self.B:d}'
        if not fname.endswith('.pkl'):
            fname += '.pkl'
        with open(BOOTSTRAP_PATH/fname, 'wb') as fd:
            pkl.dump(self.boot, fd)

    def load_bootstrap(self, fname=None, n_boots=None):
        if fname is None:
            fname = f'bootstraps_{n_boots:d}_{self.title:s}_res{self.resolution:d}_T{self.T:d}_B{self.B:d}'
        if not fname.endswith('.pkl'):
            fname += '.pkl'
        with open(BOOTSTRAP_PATH/fname, 'rb') as fd:
            self.boot = pkl.load(fd)

    def downsample(self, X0, fun=None):
        if self.torch:
            if not torch.is_tensor(X0):
                X0 = torch.tensor(X0, dtype=torch.float)
                if self.gpu:
                    X0 = X0.to('cuda')

        if self.resolution == 1:
            return X0

        if fun is None:
            fun = torch.mean if self.torch else np.mean

        T = X0.shape[1] if X0.ndim==2 else len(X0)
        if T % self.resolution != 0:
            warn(f'Data was cropped to be integer multiplication of the resolution: {T:d} -> {T - (T % self.resolution):d}')
            T -= T % self.resolution

        if self.torch:
            if X0.ndim == 1:
                X = torch.zeros(len(X0)//self.resolution, device=X0.device)
                for i in range(len(X0) // self.resolution):
                    X[i] = fun(X0[i*self.resolution : (i+1)*self.resolution])
                return X
            else:
                X = [fun(X0[:, i*self.resolution : (i+1)*self.resolution], axis=1).unsqueeze(-1) for i in range(T//self.resolution)]
                X = torch.cat(X, dim=1)
                return X
        else:
            if X0.ndim == 1:
                X = [fun(X0[i*self.resolution : (i+1)*self.resolution]) for i in range(len(X0)//self.resolution)]
                return np.array(X)
            else:
                X = [fun(X0[:, i*self.resolution : (i+1)*self.resolution], axis=1)[:,np.newaxis] for i in range(T//self.resolution)]
                X = np.concatenate(X, axis=1)
                return X

    def count_episodes(self, n):
        return n // self.T, n % self.T

    def t_test(self, X, side=None, verbose=0):
        if side is None or self.force_side: side = self.side
        X = self.downsample(X)
        n = len(X)
        x = self.calc(X, side=side)
        x0 = self.calcMean(n)
        v = self.calcVar(n)

        z = (x - x0) / np.sqrt(v)
        if verbose >= 1:
            print(dict(T=self.T, n_samples=n, n_episodes=n//self.T, n_pre=n%self.T,
                       x=x, x0=x0, v=v, sigma=np.sqrt(v), z=z))

        if side == 0:
            p = 2 * stats.norm.sf(np.abs(z))
        elif side < 0:
            p = stats.norm.sf(-z)
        else:
            p = stats.norm.sf(z)

        return p, z

    def bootstrap_store(self, n, side=None):
        self.set_seed()
        X0 = self.X0
        n_episodes, n_pre = self.count_episodes(n)

        if self.torch:
            if not torch.is_tensor(X0):
                X0 = torch.tensor(X0, dtype=torch.float)
            if self.gpu:
                X0 = X0.to('cuda')

        s = []
        for _ in range(self.B):
            if self.torch:
                X = torch.zeros(size=(0,), dtype=torch.float).to('cuda' if self.gpu else 'cpu')
                if n_episodes > 0:
                    ids = np.random.choice(np.arange(len(X0)), n_episodes, replace=True)
                    X = torch.cat((X, X0[ids,:].reshape(-1)), dim=0)
                if n_pre > 0:
                    i = np.random.choice(np.arange(len(X0)), 1, replace=True)
                    X = torch.cat((X, X0[i,:n_pre].reshape(-1)), dim=0)
            else:
                X = np.array([])
                if n_episodes > 0:
                    ids = np.random.choice(np.arange(len(X0)), n_episodes, replace=True)
                    X = np.concatenate((X, X0.take(ids, axis=0).reshape(-1)))
                if n_pre > 0:
                    i = np.random.choice(np.arange(len(X0)), 1, replace=True)
                    X = np.concatenate((X, X0[i,:n_pre].reshape(-1)))
            s.append(self.calc(X, side=side, bs=True))

        if self.gpu:
            self.clean_gpu()

        self.boot[n] = np.array(s)

    def bootstrap_test(self, X, n, side=None, return_statistic=False, bounded_zval=True, rolling=False):
        # X is assumed to be already downsampled
        # rolling (unless False) should be (rolling series key, new samples number)
        if side is None or self.force_side: side = self.side
        dist = self.boot[n]
        s = self.rolling_calc(X, rolling[0], rolling[1]//self.resolution, side=side) \
            if rolling else self.calc(X, side=side)

        bound = 1-1/(self.B+1) if bounded_zval else 1
        # p = (1 + np.sum((s - dist) <= 0)) / (1 + self.B + (side==0))
        if side > 0:
            p = (1 + np.sum(s <= dist)) / (1+self.B)
            z = -stats.norm.ppf(min(p, bound))
        elif side < 0:
            # p = 1 - p + 1/(1+self.B)
            p = (1 + np.sum(s >= dist)) / (1+self.B)
            z = stats.norm.ppf(min(p, bound))
        elif side == 0:
            # p = (1 + np.sum((s - dist) <= 0)) / (2 + self.B)
            p1 = (1 + np.sum(s <= dist)) / (1 + self.B)
            p2 = (1 + np.sum(s >= dist)) / (1 + self.B)
            z = -stats.norm.ppf(min(p1, bound))
            p = 2 * min(p1, p2)

        if return_statistic:
            return p, z, s
        return p, z

    def bootstrap_main(self, X, downsampled=False, overwrite=False, side=None, **kwargs):
        if not downsampled:
            X = self.downsample(X)
        n = len(X)
        if overwrite or n not in self.boot:
            self.bootstrap_store(n, side=side)
        return self.bootstrap_test(X, n, side=side, **kwargs)

    ### "bootstrap2*" methods below correspond to comparison between two datasets not necessarily following the reference distribution. ###

    def bootstrap2_store(self, n, side=None):
        self.set_seed()
        X0 = self.X0
        n_episodes, n_pre = self.count_episodes(n)

        if self.torch:
            raise NotImplementedError()

        s = []
        for _ in range(self.B):
            X = [np.array([]), np.array([])]
            for i in range(2):
                if n_episodes > 0:
                    ids = np.random.choice(np.arange(len(X0)), n_episodes, replace=True)
                    X[i] = np.concatenate((X[i], X0.take(ids, axis=0).reshape(-1)))
                if n_pre > 0:
                    j = np.random.choice(np.arange(len(X0)), 1, replace=True)
                    X[i] = np.concatenate((X[i], X0[j, :n_pre].reshape(-1)))
            s.append(self.calc2(X[0], X[1], side=side, bs=True))

        self.boot2[n] = np.array(s)

    def bootstrap2_test(self, X1, X2, n, side=None, return_statistic=False, bounded_zval=True):
        # X is assumed to be already downsampled
        # rolling (unless False) should be (rolling series key, new samples number)
        if side is None or self.force_side: side = self.side
        dist = self.boot2[n]
        s = self.calc2(X1, X2, side=side)

        bound = 1 - 1 / (self.B + 1) if bounded_zval else 1
        # p = (1 + np.sum((s - dist) <= 0)) / (1 + self.B + (side==0))
        if side > 0:
            p = (1 + np.sum(s <= dist)) / (1 + self.B)
            z = -stats.norm.ppf(min(p, bound))
        elif side < 0:
            # p = 1 - p + 1/(1+self.B)
            p = (1 + np.sum(s >= dist)) / (1 + self.B)
            z = stats.norm.ppf(min(p, bound))
        elif side == 0:
            # p = (1 + np.sum((s - dist) <= 0)) / (2 + self.B)
            p1 = (1 + np.sum(s <= dist)) / (1 + self.B)
            p2 = (1 + np.sum(s >= dist)) / (1 + self.B)
            z = -stats.norm.ppf(min(p1, bound))
            p = 2 * min(p1, p2)

        if return_statistic:
            return p, z, s
        return p, z

    def bootstrap2_main(self, X1, X2, downsampled=False, overwrite=False, side=None, **kwargs):
        if not downsampled:
            X1 = self.downsample(X1)
            X2 = self.downsample(X2)
        n = len(X1)
        if overwrite or n not in self.boot2:
            self.bootstrap2_store(n, side=side)
        return self.bootstrap2_test(X1, X2, n, side=side, **kwargs)


class MixTest(StatisticCalculator):
    def __init__(self, X, tests_confs, Bfac=1.5, **kwargs):
        super(MixTest, self).__init__(X, **kwargs)
        self.subside = self.side
        self.side = -1
        self.force_side = True
        self.T = self.X0.shape[1]
        self.tests = []
        self.front_test = []
        for constructor, args in tests_confs:
            if args is None: args = {}
            args = utils.update_dict(args, dict(B=int(Bfac*self.B),side=self.side), force=False)
            self.tests.append(constructor(self.X0, **args))
    def calc(self, X, record_front=None, side=None, bs=False):
        if side is None: side = self.subside
        if record_front is None: record_front = not bs
        pvals = []
        zvals = []
        for test in self.tests:
            p, z = test.bootstrap_main(X, side=side)
            pvals.append(p)
            zvals.append(z)
        i = int(np.argmin(pvals))
        if record_front:
            self.front_test.append(i)
        return pvals[i]
        # return zvals[i]
    def calc2(self, X1, X2, record_front=None, side=None, bs=False):
        if side is None or self.force_side: side = self.side
        if record_front is None: record_front = not bs
        pvals = []
        zvals = []
        for test in self.tests:
            p, z = test.bootstrap2_main(X1, X2, side=side)
            pvals.append(p)
            zvals.append(z)
        i = int(np.argmin(pvals))
        if record_front:
            self.front_test.append(i)
        return zvals[i]
    def show_front_tests(self, ax=None, ni=None, nf=None, tit=None):
        from collections import Counter
        if ax is None: ax = utils.Axes(1, axsize=(8,3))[0]
        ft = self.front_test
        if nf is not None: ft = ft[:nf]
        if ni is not None: ft = ft[ni:]
        count = Counter(ft)
        count = [(k, count[k]) for k in sorted(count.keys())]
        x = [c[0] for c in count]
        y = [c[1] for c in count]
        ax.bar(x, y)
        ax.set_xlim((-1,len(self.tests)))
        utils.labels(ax, 'test', 'times being on front', tit, fontsize=14)
        ax.set_xticks(np.arange(len(self.tests)))
        ax.set_xticklabels([t.title for t in self.tests])#, rotation=30)
    def show_fronts_per_scenario(self, scenarios, ni=0, nf=None, ax=None):
        if ax is None: ax = utils.Axes(1, axsize=(8,3))[0]
        d = pd.DataFrame()
        n = len(self.front_test) // len(scenarios)
        if nf is None: nf = n
        for i, sc in enumerate(scenarios):
            front_test = [self.tests[t].title for t in self.front_test[i*n+ni: i*n+nf]]
            d = pd.concat((d, pd.DataFrame(dict(scenario=sc, test=front_test))))
        ax = sns.countplot(data=d, hue='test', x='scenario', ax=ax)
        ax.set_ylabel('times being on front', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

################   CHILDREN CLASSES   ################
# iidMean, IndependentMean, SimpleMean, WeightedMean, CVaR

class iidMean(StatisticCalculator):
    def __init__(self, X, **kwargs):
        super(iidMean, self).__init__(X, **kwargs)
        X = self.X0.reshape(-1)
        self.mean = np.mean(X)
        self.var = np.var(X)
    def calc(self, X, side=None, bs=False):
        return np.mean(X)
    def calcMean(self, n):
        return self.mean
    def calcVar(self, n):
        return self.var / n

class IndependentMean(StatisticCalculator):
    def __init__(self, X, **kwargs):
        super(IndependentMean, self).__init__(X, **kwargs)
        means = np.mean(self.X0, axis=0)
        vars = np.var(self.X0, axis=0)
        self.T = len(means)
        self.cum_reward = np.concatenate(([0], np.cumsum(means)))
        self.Vars = np.concatenate(([0], np.cumsum(vars)))
    def calc(self, X, side=None, bs=False):
        return np.mean(X)
    def calcMean(self, n):
        n_episodes, n_pre = self.count_episodes(n)
        return (n_episodes * self.cum_reward[self.T] + self.cum_reward[n_pre]) / n
    def calcVar(self, n):
        n_episodes = n // self.T
        n_pre = n % self.T
        return (n_episodes * self.Vars[self.T] + self.Vars[n_pre]) / n**2

class SimpleMean(StatisticCalculator):
    def __init__(self, X, **kwargs):
        super(SimpleMean, self).__init__(X, **kwargs)
        means = np.mean(self.X0, axis=0)
        self.G = GramSchmidtVars(np.cov(self.X0.transpose()))
        self.T = len(self.G)
        self.cum_reward = np.concatenate(([0], np.cumsum(means)))
        self.Vars = np.array([np.sum(np.sum(self.G[:i,:i], axis=0)**2)
                              for i in range(len(self.G)+1)])
    def get_temporal_weights(self):
        return np.ones(self.T) / self.T
    def calc(self, X, side=None, bs=False):
        return np.mean(X)
    def calcMean(self, n):
        n_episodes, n_pre = self.count_episodes(n)
        return (n_episodes * self.cum_reward[self.T] + self.cum_reward[n_pre]) / n
    def calcVar(self, n):
        n_episodes, n_pre = self.count_episodes(n)
        return (n_episodes * self.Vars[self.T] + self.Vars[n_pre]) / n**2
    # def rolling_calc(self, X, key):
    #     n = len(X)
    #     n_episodes, n_pre = self.count_episodes(n)
    #     if n_pre == 0:
    #         # end of episode - update rolling statistic
    #         new_stat = self.calc(X[-self.T:])
    #         self.rolling_statistic[key] = self.rolling_statistic[key][1:] + [new_stat]
    #         return np.mean(self.rolling_statistic[key])
    #         pass
    #     else:
    #         pass

class CUSUM(StatisticCalculator):
    def __init__(self, X, weights=2, K=0.5, **kwargs):
        super(CUSUM, self).__init__(X, **kwargs)
        self.K = K
        self.means = np.mean(self.X0, axis=0)
        self.T = len(self.means)
        self.stds = np.std(self.X0, axis=0)
        if weights == 0:
            self.weights = np.ones(self.T)
        elif weights == 1:
            self.weights = np.ones(self.T) / np.std(self.X0)
        else:
            self.weights = 1/self.stds

    def get_temporal_weights(self):
        return self.weights

    def rolling_calc(self, X, key, new_samples=1, **kwargs):
        if key not in self.rolling_statistic:
            self.rolling_statistic[key] = self.calc(X, **kwargs)
        else:
            self.rolling_statistic[key] = self.calc(X=X, t0=len(X)-new_samples, s=self.rolling_statistic[key], **kwargs)
        return self.rolling_statistic[key]

    def calc(self, X, t0=0, s=0, side=None, bs=False):
        n = len(X)
        n_episodes, n_pre = self.count_episodes(n)

        for i in range(n_episodes):
            for j in range(self.T):
                if i*self.T+j < t0:
                    continue
                if self.side < 0:
                    s = min(0, s + self.weights[j]*(X[i*self.T+j]-self.means[j]) + self.K)
                else:
                    s = max(0, s + self.weights[j]*(X[i*self.T+j]-self.means[j]) - self.K)

        if n_pre > 0:
            for j in range(n_pre):
                if n_episodes*self.T+j < t0:
                    continue
                if self.side < 0:
                    s = min(0, s + self.weights[j]*(X[n_episodes*self.T+j]-self.means[j]) + self.K)
                else:
                    s = max(0, s + self.weights[j]*(X[n_episodes*self.T+j]-self.means[j]) - self.K)

        return s

class IndependentWeightedMean(StatisticCalculator):
    def __init__(self, X, squared_weights=True, **kwargs):
        super(IndependentWeightedMean, self).__init__(X, **kwargs)
        self.means = np.mean(self.X0, axis=0)
        self.vars = np.var(self.X0, axis=0)
        self.weights = 1/self.vars if squared_weights else 1/np.sqrt(self.vars)
        self.weights /= np.sum(self.weights)
        self.cum_weights = np.cumsum(self.weights)
        self.T = len(self.vars)

    def get_temporal_weights(self):
        return self.weights

    def calc(self, X, side=None, bs=False):
        n = len(X)
        n_episodes, n_pre = self.count_episodes(n)

        s = 0
        for i in range(n_episodes):
            s += np.sum(self.weights * (X[i*self.T:(i+1)*self.T]))
        if n_pre > 0:
            s += np.sum(self.weights[:n_pre] * X[n_episodes*self.T:])

        normalizer = n_episodes
        if n_pre > 0:
            normalizer += self.cum_weights[n_pre] / self.cum_weights[-1]
        return s / normalizer

class Hotelling(StatisticCalculator):
    def __init__(self, X, approx_partS=False, **kwargs):
        super(Hotelling, self).__init__(X, **kwargs)
        self.side = 1 # x^2 -> changes are always positive
        self.force_side = True
        self.approx_partS = approx_partS
        self.means = np.mean(self.X0, axis=0)
        self.Sigma = np.cov(self.X0.transpose())
        self.T = len(self.Sigma)
        S1 = np.linalg.inv(self.Sigma)
        self.fullSigma = S1
        self.partSigmas = {}

    def calc(self, X, side=None, bs=False):
        n = len(X)
        n_episodes, n_pre = self.count_episodes(n)

        s = 0
        for i in range(n_episodes):
            x = X[i*self.T:(i+1)*self.T] - self.means
            s += np.matmul(np.matmul(x.transpose(), self.fullSigma), x)
        if n_pre > 0:
            x = X[n_episodes*self.T:] - self.means[:n_pre]
            if self.approx_partS:
                S = self.fullSigma[:n_pre,:n_pre]
            else:
                if n_pre not in self.partSigmas:
                    S1 = np.linalg.inv(self.Sigma[:n_pre, :n_pre])
                    self.partSigmas[n_pre] = S1
                S = self.partSigmas[n_pre]
            s += np.matmul(np.matmul(x.transpose(), S), x)

        normalizer = n_episodes + n_pre/self.T
        return s / normalizer

class WeightedMean(StatisticCalculator):
    def __init__(self, X, approx_partW=False, force_positive=False, **kwargs):
        super(WeightedMean, self).__init__(X, **kwargs)
        self.approx_partW = approx_partW
        self.force_positive = force_positive
        self.Sigma = np.cov(self.X0.transpose())
        self.T = len(self.Sigma)
        S1 = np.linalg.inv(self.Sigma)
        self.fullSigma = S1# + S1.transpose()
        self.fullW = np.sum(self.fullSigma, axis=0)
        if self.force_positive:
            # print(f'Negative weights: {np.sum(self.fullW<0):.0f}/{self.T}')
            self.fullW = np.clip(self.fullW, 0, None)
        self.sumW = np.sum(self.fullW)
        self.fullW /= self.sumW
        self.partSigmas = {}
        self.partW = {}

    def get_temporal_weights(self):
        return self.fullW

    def calc(self, X, side=None, bs=False):
        n = len(X)
        n_episodes, n_pre = self.count_episodes(n)

        s = 0
        for i in range(n_episodes):
            s += np.sum(self.fullW * X[i * self.T:(i + 1) * self.T])
        if n_pre > 0:
            if self.approx_partW:
                W = self.fullW[:n_pre]
            else:
                if n_pre not in self.partSigmas:
                    S1 = np.linalg.inv(self.Sigma[:n_pre, :n_pre])
                    self.partSigmas[n_pre] = S1# + S1.transpose()
                    self.partW[n_pre] = np.sum(self.partSigmas[n_pre], axis=0)
                    if self.force_positive:
                        self.partW[n_pre] = np.clip(self.partW[n_pre], 0, None)
                    self.partW[n_pre] /= self.sumW
                W = self.partW[n_pre]
            s += np.sum(W * X[n_episodes * self.T:])

        normalizer = n_episodes
        if n_pre > 0:
            normalizer += np.sum(W)
        return s / normalizer

class WeightedCUSUM(StatisticCalculator):
    def __init__(self, X, approx_partW=False, K=0.5, side=-1, **kwargs):
        super(WeightedCUSUM, self).__init__(X, **kwargs)
        self.side = side
        self.K = K
        self.means = np.mean(self.X0, axis=0)
        self.Sigma = np.cov(self.X0.transpose())
        self.approx_partW = approx_partW
        self.T = len(self.Sigma)
        S1 = np.linalg.inv(self.Sigma)
        self.fullSigma = S1 + S1.transpose()
        self.fullW = np.sum(self.fullSigma, axis=0)
        self.sumW = np.sum(self.fullW)
        self.fullW /= self.sumW
        self.partSigmas = {}
        self.partW = {}

    def get_temporal_weights(self):
        return self.fullW

    def calc(self, X, side=None, bs=False):
        n = len(X)
        n_episodes, n_pre = self.count_episodes(n)

        s = 0
        for i in range(n_episodes):
            for j in range(self.T):
                if self.side < 0:
                    s = min(0, s + self.fullW[j]*(X[i*self.T+j]-self.means[j]) + self.K)
                else:
                    s = max(0, s + self.fullW[j]*(X[i*self.T+j]-self.means[j]) - self.K)

        if n_pre > 0:
            if self.approx_partW:
                W = self.fullW
            else:
                if n_pre not in self.partSigmas:
                    S1 = np.linalg.inv(self.Sigma[:n_pre,:n_pre])
                    self.partSigmas[n_pre] = S1 + S1.transpose()
                    self.partW[n_pre] = np.sum(self.partSigmas[n_pre], axis=0)
                    self.partW[n_pre] /= self.sumW
                W = self.partW[n_pre]
            for j in range(n_pre):
                if self.side < 0:
                    s = min(0, s + W[j]*(X[n_episodes*self.T+j]-self.means[j]) + self.K)
                else:
                    s = max(0, s + W[j]*(X[n_episodes*self.T+j]-self.means[j]) - self.K)

        return s

class CVaR(StatisticCalculator):
    def __init__(self, X, p=0.1, normalized=1, **kwargs):
        super(CVaR, self).__init__(X, **kwargs)
        self.means = np.mean(self.X0, axis=0)
        self.normalized = normalized
        self.std = np.std(self.X0, axis=0)
        self.norm = self.std**self.normalized if self.normalized else np.ones_like(self.std)
        self.T = len(self.means)
        self.p = p
        self.chosen_indices = []

    def calc(self, X, side=None, record_ids=None, bs=False):
        if side is None or self.force_side: side = self.side
        if record_ids is None: record_ids = not bs
        n = len(X)
        n_episodes, n_pre = self.count_episodes(n)

        # Z = X - means
        Z = np.zeros_like(X)
        for i in range(n_episodes):
            Z[i*self.T : (i+1)*self.T] = X[i*self.T : (i+1)*self.T] - self.means
            if self.normalized:
                Z[i*self.T: (i+1)*self.T] /= self.norm
        if n_pre > 0:
            Z[n_episodes*self.T:] = X[n_episodes*self.T:] - self.means[:n_pre]
            if self.normalized:
                Z[n_episodes*self.T:] /= self.norm[:n_pre]

        # CVaR
        n_risk = int(np.round(self.p * len(X)))
        if n_risk == 0:
            n_risk = 1
        if record_ids:
            ids = Z.argsort()
            if side > 0: ids = ids[::-1]
            self.chosen_indices.append(ids[:n_risk])
        return np.mean(sorted(Z, reverse=(side>0))[:n_risk])

    def show_indices(self, axs=None, ni=None, nf=None, tit=None):
        from collections import Counter
        if axs is None: axs = utils.Axes(2, axsize=(8, 3))

        ax = axs[0]
        ax.plot(np.arange(len(self.norm)), 1/self.norm, '.-')
        utils.labels(ax, 't', 'weights', tit, fontsize=16)

        ax = axs[1]
        ids = self.chosen_indices
        if nf is not None: ids = ids[:nf]
        if ni is not None: ids = ids[ni:]
        ids = [i for ep in ids for i in ep]
        count = Counter(ids)
        count = [(k, count[k]) for k in sorted(count.keys())]
        x = [c[0] for c in count]
        y = [c[1] for c in count]
        ax.plot(x, y, '.-')
        ax.set_xlim((-1, self.T))
        utils.labels(ax, 't', 'count of shame', tit, fontsize=16)


class TransformedCVaR(StatisticCalculator):
    def __init__(self, X, p=0.1, approx_partS=False, force_positive=False, zeroize_corr_below=0, bug=False, **kwargs):
        super(TransformedCVaR, self).__init__(X, **kwargs)
        self.inherent_side = self.side
        self.side = -1
        self.force_side = True
        self.bug = bug
        self.force_positive = force_positive
        self.corr_thresh = zeroize_corr_below
        self.means = np.mean(self.X0, axis=0)
        self.approx_partS = approx_partS
        self.Sigma = np.cov(self.X0.transpose())
        self.T = len(self.Sigma)
        if self.corr_thresh > 0:
            std = np.std(self.X0, axis=0)
            corr = np.array([[self.Sigma[i,j]/(std[i]*std[j])
                              for j in range(self.T)] for i in range(self.T)])
            self.Sigma[np.abs(corr)<self.corr_thresh] = 0
        SigInv = np.linalg.inv(self.Sigma)
        self.SigInvDiag = np.diag(SigInv)
        if self.force_positive:
            # print(np.sum(SigInv<0), np.sum(SigInv<np.inf))
            SigInv = np.clip(SigInv, 0, None)
        self.S = SigInv + self.bug * SigInv.transpose()
        self.partSigInvDiag = {}
        self.partS = {}
        self.p = p
        self.n_risk = int(np.ceil(self.p * self.T))
        self.chosen_indices = []

    # def get_temporal_weights(self, X):
    #     return self.fullW

    def calc(self, X, side=None, record_ids=None, bs=False, means=None):
        if side is None: side = self.inherent_side
        if record_ids is None: record_ids = not bs
        if means is None: means = self.means
        means = np.array(means)
        if means.ndim == 0:
            means = np.repeat(means, self.T)
        n = len(X)
        n_episodes, n_pre = self.count_episodes(n)

        # Z = X - mu
        Z = np.zeros_like(X)
        for i in range(n_episodes):
            Z[i*self.T : (i+1)*self.T] = X[i*self.T : (i+1)*self.T] - means
        if n_pre > 0:
            Z[n_episodes*self.T:] = X[n_episodes*self.T:] - means[:n_pre]

        # s = S*Z
        s = np.zeros(self.T)
        for i in range(n_episodes):
            SigInvDiag = self.SigInvDiag
            S = self.S
            s += np.matmul(S, Z[i*self.T:(i+1)*self.T]) - self.bug * SigInvDiag
        if n_pre > 0:
            if self.approx_partS:
                SigInvDiag = self.SigInvDiag[:n_pre]
                S = self.S[:n_pre, :n_pre]
            else:
                if n_pre not in self.partS:
                    SigInv = np.linalg.inv(self.Sigma[:n_pre, :n_pre])
                    self.partSigInvDiag[n_pre] = np.diag(SigInv)
                    if self.force_positive:
                        SigInv = np.clip(SigInv, 0, None)
                    self.partS[n_pre] = SigInv + self.bug * SigInv.transpose()
                SigInvDiag = self.partSigInvDiag[n_pre]
                S = self.partS[n_pre]
            s[:n_pre] += np.matmul(S, Z[n_episodes*self.T:]) - self.bug * SigInvDiag

        normalizer = self.n_risk * (n_episodes + n_pre / self.T)
        s /= normalizer

        # sort s and take top elements
        ids = s.argsort()
        s = sorted(s)
        if side < 0:
            if record_ids:
                self.chosen_indices.append(ids[:self.n_risk])
            s = np.sum(s[:self.n_risk])
        elif side > 0:
            if record_ids:
                self.chosen_indices.append(ids[-self.n_risk:])
            s = -np.sum(s[-self.n_risk:])
        else: # side==0
            s1 = np.sum(s[:self.n_risk])
            s2 = -np.sum(s[-self.n_risk:])
            if record_ids:
                if s1 < s2:
                    self.chosen_indices.append(ids[:self.n_risk])
                else:
                    self.chosen_indices.append(ids[-self.n_risk:])
            s = min(s1,s2)

        return s

    def calc2(self, X1, X2, side=None, bs=False, **kwargs):
        return self.calc(X2-X1, side=side, bs=bs, means=0, **kwargs)

    def show_indices(self, axs=None, ni=None, nf=None, tit=None):
        from collections import Counter
        if axs is None: axs = utils.Axes(2, axsize=(8,3))

        ax = axs[0]
        ax.plot(np.arange(len(self.S)), self.S.sum(axis=0), '.-')
        utils.labels(ax, 't', 'uniform weight', tit, fontsize=16)

        ax = axs[1]
        ids = self.chosen_indices
        if nf is not None: ids = ids[:nf]
        if ni is not None: ids = ids[ni:]
        ids = [i for ep in ids for i in ep]
        count = Counter(ids)
        count = [(k, count[k]) for k in sorted(count.keys())]
        x = [c[0] for c in count]
        y = [c[1] for c in count]
        ax.plot(x, y, '.-')
        ax.set_xlim((-1,self.T))
        utils.labels(ax, 't', 'count of shame', tit, fontsize=16)
