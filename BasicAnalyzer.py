
'''
This module defines the Analyzer object, that stores episodic time-series and analyze them for differences.

INPUT:
The constructor requires a path to a pickle file with a dictionary whose entries are:
    - scenarios: a dataframe with the columns scenario/episode/seed/score, where each row corresponds to a single episode.
    - env_args: env_args[scenario_name] = a dictionary of the arguments used to generate episodes in this scenario.
    - data: data[scenario_name][i] = a list of values (e.g. rewards) in the i'th episode of the scenario. all episodes are assumed to be of the same length.

MODULE STRUCTURE:
    - General methods
    - Methods corresponding to descriptive statistics on episodes level (not looking within episodes)
    - Methods corresponding to statistical tests on episode level (not looking within episodes)
    - Methods corresponding to descriptive statistics on time-steps level
    - Methods corresponding to statistical tests on time-steps level
    - Methods corresponding to sequential tests
    - Methods corresponding to summary: run_summarizing_tests() & analysis_summary()

Example:
    # Define statistical tests (dictionary of pairs (constructor,args)):
    tests = dict(
        Mean = (Stats.SimpleMean, dict()),
        CUSUM = (Stats.CUSUM, dict()),
        UDT = (Stats.WeightedMean, dict()),
        PDT = (Stats.TransformedCVaR, dict(p=0.9)),
    )
    # Create an analyzer of HalfCheetah rewards, with sample-resolution of 25 (i.e. each sample is the mean of 25 time-steps):
    A = Analyzer.Analyzer('HalfCheetah_data', Path('Spectation/data/HalfCheetah-v3'), resolution=25, title='HalfCheetah')
    # Run both individual & sequential tests (levels=1 would run only individual tests):
    A.run_summarizing_tests(save_name='HalfCheetah', ns=(100,300,1000,3000,10000,30000,50000),
        default_tester_args=dict(B=5000), lookback_horizons=(5,50), seq_test_len=50, tests=tests, levels=3)
    # Analyze both individual & sequential tests results:
    A.analysis_summary('HalfCheetah', do_sequential=True, do_non_seq=True)

Code profiling:
- downsampling is the costliest action in all the code.
  it was moved to Analyzer (from StatsCalculator) to reduce number of calls (reduced running time ~x10).
- transforming p to z is about half as costly as calculating the statistic, and 50% of these calls were redundant.
- never assign single values to df.loc[index, col]. do the assignment for all the column together.
- next beneficial change is an efficient implementation of the rolling statistics calculation.

Ido Greenberg, 2020
'''

import pickle as pkl
from warnings import warn
from pathlib import Path
from time import time
from collections import Counter

import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf, pacf

import StatsCalculator as Stats
import utils

pd.options.mode.chained_assignment = None  # default='warn'
DATA = Path('Spectation/data/HalfCheetah-v3')

class Analyzer:
    def __init__(self, fname, path=DATA, side=-1, resolution=1, seed=1, title='', variance_analysis=None,
                 uniform_T=True):
        if not fname.endswith('.pkl'):
            fname += '.pkl'
        with open(path/fname, 'rb') as fd:
            tmp = pkl.load(fd)
        meta, env_args, data = tmp['scenarios'], tmp['env_args'], tmp['data']
        self.m = meta
        self.m['meta_scenario'] = [sc[:-3] if len(sc) > 3 else sc for sc in self.m.scenario]
        self.env_args = env_args
        self.resolution = resolution
        to_ndarray = lambda a,N: np.array(a)
        T = None
        if not uniform_T:
            def maxT(dd):
                return max([max([len(aa) for aa in a]) for a in dd.values()])
            def to_ndarray(a, N=None):
                n1 = len(a)
                n2 = max([len(aa) for aa in a]) if N is None else N
                y = np.zeros((n1, n2))
                for i,aa in enumerate(a):
                    y[i, :len(aa)] = aa
                return y
            T = maxT(data)
        self.d = {s:to_ndarray(d,T) for s,d in data.items()} # N x T
        self.downsample()
        self.title = title
        self.seed = seed
        self.side = side
        self.scenarios = np.unique(self.m.scenario)
        self.T = len(self.d[self.scenarios[0]][0])

        self.scores_std = {s:np.std(self.get_scenario_scores(s)) for s in self.scenarios}
        self.G0 = {s:np.cov(np.array(self.d[s]).transpose()) for s in self.scenarios}

        self.boot = {}
        self.episode_stats = {}
        if variance_analysis:
            if isinstance(variance_analysis, str):
                self.episode_variance_analysis(scenario=variance_analysis)
            else:
                self.episode_variance_analysis()

        self.res = {}
        # self.alpha = {} # (alpha0, horizons, frequency) -> alpha

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        np.random.seed(seed)

    def get_scenario_scores(self, scenario):
        return self.m.score[self.m.scenario == scenario]

    def get_meta_scenarios_summary(self, verbose=1):
        ret = self.m.groupby('meta_scenario').apply(
            lambda m: f'{len(m):d} episodes, {len(np.unique(m.scenario)):d} blocks, {len(np.unique(m.seed)):d} unique seeds.'
            # dict(tot_episodes=len(m), sub_scenarios=len(np.unique(m.scenario)),
            #                unique_seeds=len(np.unique(m.seed)))
        )
        if verbose >= 1:
            print(ret)
        return ret

    def n_episodes(self, scenario):
        return (self.m.scenario == scenario).sum()

    def get_scenarios_data(self, scenario_prefix):
        return np.concatenate([self.d[s] for s in self.scenarios if s.startswith(scenario_prefix)], axis=0)

    def z2p(self, z, side=None):
        if side is None: side = self.side
        if side == 0:
            p = 2 * stats.norm.sf(np.abs(z))
        elif side < 0:
            p = stats.norm.sf(-z)
        else:
            p = stats.norm.sf(z)
        return p

    def smooth_scenario_data(self, scenario='ref', resolution=1, fun=np.sum):
        assert (self.T % resolution == 0)
        if resolution == 1:
            return self.d[scenario]
        return np.array([fun(a, axis=1) for a in np.split(self.d[scenario], self.T//resolution, axis=1)]).transpose()

    def downsample(self, fun=None):
        if self.resolution == 1:
            return

        if fun is None: fun = np.mean

        for nm, X in self.d.items():
            T = X.shape[1]
            if T % self.resolution != 0:
                warn(f'Data was cropped to be integer multiplication of the resolution: {T:d} -> {T - (T % self.resolution):d}')
                T -= T % self.resolution
            X = [fun(X[:, i*self.resolution : (i+1)*self.resolution], axis=1)[:,np.newaxis] for i in range(T//self.resolution)]
            X = np.concatenate(X, axis=1)
            self.d[nm] = X

    def plot_episode_transitions(self, ax, Tf, Ti=0, color='k', linestyle=':', linewidth=0.5):
        ax.grid(False, axis='x')
        for t in range(Ti, Tf+1, self.T * self.resolution):
            ax.axvline(t, color=color, linestyle=linestyle, linewidth=linewidth)

    def merge_analyzer(self, A2, pre1='', pre2='', scenarios2=None):
        if pre1 and not pre1.endswith('_'): pre1 += '_'
        if pre2 and not pre2.endswith('_'): pre2 += '_'
        if scenarios2 is None: scenarios2 = A2.scenarios

        if pre1:
            self.m['scenario'] = [pre1+s for s in self.m['scenario']]
            self.m['meta_scenario'] = [sc[:-3] if not sc.endswith('ref') else sc for sc in self.m.scenario]
            self.d = {(pre1+k): v for k, v in self.d.items()}
            self.scenarios = [pre1+s for s in self.scenarios]
            self.env_args = {(pre1+k): v for k, v in self.env_args.items()}

        if pre2:
            A2.m['scenario'] = [pre2+s for s in A2.m['scenario']]
            A2.m['meta_scenario'] = [sc[:-3] if not sc.endswith('ref') else sc for sc in A2.m.scenario]
            A2.d = {(pre2+k): v for k, v in A2.d.items()}
            A2.scenarios = [pre2+s for s in A2.scenarios]
            A2.env_args = {(pre2+k): v for k, v in A2.env_args.items()}
            scenarios2 = [pre2+s for s in scenarios2]

        self.m = pd.concat((self.m, A2.m[A2.m.scenario.isin(scenarios2)]))
        self.d = utils.update_dict(self.d, {s:v for s,v in A2.d.items() if s in scenarios2})
        self.scenarios = np.concatenate((self.scenarios, scenarios2))
        self.env_args = utils.update_dict(self.env_args, {s:v for s,v in A2.env_args.items() if s in scenarios2})

    def rename_scenarios(self, meta_scenarios_map):
        for sc1, sc2 in meta_scenarios_map.items():
            if sc1 == sc2:
                continue
            blocks = np.unique(self.m[self.m.meta_scenario==sc1].scenario)
            self.m.loc[self.m.meta_scenario==sc1,'scenario'] = [s.replace(sc1,sc2)
                                                                for s in self.m[self.m.meta_scenario==sc1].scenario]
            self.m.loc[self.m.meta_scenario==sc1,'meta_scenario'] = sc2
            for sc in blocks:
                self.d[sc.replace(sc1,sc2)] = self.d[sc]
                self.d[sc] = None
                self.env_args[sc.replace(sc1, sc2)] = self.env_args[sc]
                self.env_args[sc] = None
            self.scenarios = np.unique(self.m.scenario)

    ############   EPISODES: DESCRIPTIVE STATISTICS   ############

    def scores_boxplot(self, scenarios=None, ax=None, figsize=(12, 4), rotation=20):
        if scenarios is None:
            scenarios = self.scenarios
        ax = utils.Axes(1, axsize=figsize)[0] if ax is None else ax
        ax = sns.boxplot(data=self.m[self.m.scenario.isin(scenarios)], x='scenario', y='score', showmeans=True, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
        return ax

    @staticmethod
    def qqplot_normal(x, tit=None, ax=None, labs=None):
        x = np.array(x)
        if x.ndim == 1:
            x = np.array([x])
        if labs is None:
            labs = len(x) * [None]

        ax = utils.Axes(1)[0] if ax is None else ax
        ax.plot([np.min(x.reshape(-1)), np.max(x.reshape(-1))],
                [np.min(x.reshape(-1)), np.max(x.reshape(-1))], 'k--')

        for xx, l in zip(x,labs):
            z, y = stats.probplot(xx, dist="norm")[0]
            ax.plot(z, y, '.-', label=l)
        if labs[0]:
            ax.legend(fontsize=10)
        utils.labels(ax, 'Theoretical quantiles', 'Ordered values', tit, 14)

    @staticmethod
    def qqplot_unif(x, a=0, b=1, xlab='', ylab=None, tit=None, labs=None, ax=None):
        x = np.array(x)
        if x.ndim == 1:
            x = np.array([x])
        if labs is None:
            labs = len(x) * [None]

        ax = utils.Axes(1)[0] if ax is None else ax
        ax.plot([0, 100], [a, b], 'k--')
        for xx, l in zip(x,labs):
            ax.plot(utils.get_quantiles(xx)[0], label=l)
        utils.labels(ax, f'{xlab} quantile [%]', ylab, tit, 13)
        if labs[0]:
            ax.legend(fontsize=10)
        return ax

    @staticmethod
    def qqplot(x1, x2, lab1='', lab2='', labs=None, ax=None, figsize=(6, 6)):
        x2 = np.array(x2)
        if x2.ndim == 1:
            x2 = np.array([x2])
        if labs is None:
            labs = len(x2) * [None]

        ax = utils.Axes(1, axsize=figsize)[0] if ax is None else ax
        a = min(np.min(x1), np.min(x2.reshape(-1)))
        b = max(np.max(x1), np.max(x2.reshape(-1)))
        ax.plot([a, b], [a, b], 'k--')
        for xx, l in zip(x2, labs):
            ax.plot(utils.get_quantiles(x1)[0], utils.get_quantiles(xx)[0], '.', label=l)
        if labs[0]: ax.legend(fontsize=10)
        if lab1: ax.set_xlabel(lab1)
        if lab2: ax.set_ylabel(lab2)
        return ax

    def scenarios_qplots(self, scenarios=None, ax=None):
        if scenarios is None:
            scenarios = self.scenarios
        ax = utils.Axes(1)[0] if ax is None else ax
        for s in scenarios:
            utils.plot_quantiles(self.get_scenario_scores(s), ax=ax, label=s)
        utils.labels(ax, 'Quantile [%]', 'Scenario score')
        ax.legend(fontsize=10)
        return ax

    def episode_scores_descriptive_summary(self, ref=None, scenarios=None, z=None, axs=None,
                                           n_plot=24, rotation=90, ylim=(6.2,7.2)):
        if scenarios is None:
            scenarios = self.scenarios
        if ref is None:
            ref = scenarios[0]
        scenarios2 = [s for s in scenarios if s != ref]
        if z is None:
            z = np.random.random(len(scenarios2))
        if axs is None:
            axs = utils.Axes(6, axsize=(8,4))

        a = 0

        # scores boxplot
        self.scores_boxplot(scenarios=scenarios[:n_plot], ax=axs[a], rotation=rotation)
        a += 1

        # number of episodes per scenario
        axs[a].hist([self.n_episodes(s) for s in scenarios])
        utils.labels(axs[a], '#episodes', title='Histogram')
        a += 1

        # score vs. episode-quantile by scenario
        tmp = [ref, scenarios[np.argmin(z)]] + list(np.random.choice(scenarios2, 2, replace=False)) + [
            scenarios[np.argmax(z)]]
        self.scenarios_qplots(tmp, axs[a])
        self.scenarios_qplots(tmp, axs[a+1])
        axs[a+1].set_ylim(ylim)
        a += 2

        # qqplot scenarios vs. ref
        ax = axs[a]
        x1 = self.get_scenario_scores(ref)
        x2 = [self.get_scenario_scores(s) for s in tmp[1:]]
        Analyzer.qqplot(x1, x2, lab1=f'{ref} quantiles', lab2='new quantiles', labs=tmp[1:], ax=ax)
        a += 1

        # qqplot scenarios vs. ref (zoom-in)
        ax = axs[a]
        x1 = self.get_scenario_scores(ref)
        x2 = [self.get_scenario_scores(s) for s in tmp[1:]]
        Analyzer.qqplot(x1, x2, lab1=f'{ref} quantiles', lab2='new quantiles', labs=tmp[1:], ax=ax)
        ax.set_xlim(ylim)
        ax.set_ylim(ylim)
        a += 1

        return axs

    ############   EPISODES: STATISTICAL TESTS   ############

    def compare_means_t_test(self, s2, s1, side=None, verbose=0):
        if side is None: side = self.side

        x1 = self.get_scenario_scores(s1)
        x2 = self.get_scenario_scores(s2)
        n1 = len(x1)
        n2 = len(x2)
        m1 = np.mean(x1)
        m2 = np.mean(x2)
        v1 = np.var(x1) / n1
        v2 = np.var(x2) / n2

        if verbose >= 2:
            print(dict(n1=n1, n2=n2, m1=m1, m2=m2, v1=v1, v2=v2))

        z = (m2 - m1) / np.sqrt(v1 + v2)
        p = self.z2p(z, side)


        if verbose >= 1 or n1 < 30 or n2 < 30:
            warn(f"T-test is actually implemented as Z-test for the sake of lazyness, but should be fine (n1={n1:.0f}, n2={n2:.0f})")

        return p, z

    def bootstrap(self, s, B, n=None, fun=np.mean, fun_name='mean', Tf=None, overwrite=False):
        if Tf is None: Tf = self.T
        if n is None:
            n = self.n_episodes(s)
        if fun_name not in self.boot:
            self.boot[fun_name] = {}

        if s not in self.boot[fun_name]:
            self.boot[fun_name][s] = {}
        if n not in self.boot[fun_name][s]:
            self.boot[fun_name][s][n] = {}

        if Tf not in self.boot[fun_name][s][n] or overwrite:
            if Tf == self.T:
                x = self.get_scenario_scores(s)
                st = np.array([fun(np.random.choice(x, n, replace=True)) for _ in range(B)])
            else:
                if fun is not np.mean:
                    raise ValueError('Partial-episode bootstrap only supports mean-reducer.')
                x = self.d[s]
                full_scores = np.mean(x, axis=1)
                pre_scores = np.mean(x[:, :Tf], axis=1)
                n_samples = (n-1) * self.T + Tf
                w_pre = Tf / n_samples
                st = np.array([
                    w_pre * np.mean(np.random.choice(pre_scores, 1, replace=True)) + \
                    (1-w_pre) * (np.mean(np.random.choice(full_scores, n-1, replace=True)) if n>1 else 0)
                    for _ in range(B)
                ])
            self.boot[fun_name][s][n][Tf] = st

    def mean_bootstrap_test(self, s, s0='ref', B=2000, fun=np.mean, fun_name='mean',
                            episodes=None, Tf=None, side=None, seed=None, overwrite=False):
        if Tf is None: Tf = self.T
        if side is None: side = self.side
        self.set_seed(seed)

        # get statistic
        # if episodes is None and Tf is None:
        n_episodes = self.n_episodes(s)
        x = self.get_scenario_scores(s)
        st = fun(x)
        # else:
        #     st, n_episodes, Tf, n_samples = self.partial_scenario_stats(s, episodes, Tf, fun=fun)

        # get reference distribution
        self.bootstrap(s0, B=B, n=n_episodes, fun=fun, fun_name=fun_name, Tf=Tf, overwrite=overwrite)

        # get bootstrap p-val
        p = (1 + np.sum((st - self.boot[fun_name][s0][n_episodes][Tf]) <= 0)) / (1 + B + (side==0))
        z = -stats.norm.ppf(p)
        if side < 0:
            p = 1 - p + 1/(1+B)
            z = stats.norm.ppf(p)
        elif side == 0:
            p = 2 * min(p, 1 - p)
        return p, z

    def get_pvals_vs_ref(self, sref='ref', scenarios=None, test=None, **kwargs):
        if scenarios is None:
            scenarios = self.scenarios
        scenarios = [s for s in scenarios if s != sref]
        if test is None:
            test = self.compare_means_t_test
        out = []
        for s in scenarios:
            out.append(test(s, sref, **kwargs))
        return out

    def get_pvals_pairs(self, scenarios=None, test=None, **kwargs):
        if scenarios is None:
            scenarios = self.scenarios
        if test is None:
            test = self.compare_means_t_test
        n = len(scenarios)
        p = np.zeros((n, n))
        for i, s1 in enumerate(scenarios):
            for j, s2 in enumerate(scenarios):
                p[s1, s2] = test(s2, s1, **kwargs)[0]
        return p

    def episode_scores_tests_summary(self, scenarios2, z1, z2, p1, p2, lab1='T-test', lab2='Bootstrap',
                                     axs=None, n_plot=24, rotation=90):
        if axs is None:
            axs = utils.Axes(4, axsize=(8,4))

        a = 0

        ax = axs[a]
        tmp = pd.DataFrame(dict(
            scenario=scenarios2[:n_plot] + scenarios2[:n_plot],
            test=np.repeat((lab1, lab2), min(len(z1), n_plot)),
            zval=np.concatenate((z1[:n_plot], z2[:n_plot]))
        ))
        print(tmp)
        sns.barplot(data=tmp, x='scenario', y='zval', hue='test', ax=ax)
        ax.grid()
        ax.set_xticklabels(scenarios2[:n_plot], rotation=rotation)
        a += 1

        Analyzer.qqplot_normal((z1, z2), 'Z-values', axs[a],
                               (f'{lab1:s} ({np.min(z1):.1f} - {np.max(z1):.1f})',
                                f'{lab2:s} ({np.min(z2):.1f} - {np.max(z2):.1f})'))
        a += 1

        Analyzer.qqplot_unif((p1, p2), a=0, b=1, xlab='Scenario', ylab='P-value', labs=(lab1, lab2), ax=axs[a])
        a += 1

        Analyzer.qqplot_unif((p1, p2), a=0, b=1, xlab='Scenario', ylab='P-value', ax=axs[a],
                             labs=(f'{lab1:s}: P(p<0.01)={np.mean(np.array(p1)<0.01):.3f}, P(p<0.05)={np.mean(np.array(p1)<0.05):.3f}',
                                   f'{lab2:s}: P(p<0.01)={np.mean(np.array(p2)<0.01):.3f}, P(p<0.05)={np.mean(np.array(p2)<0.05):.3f}'))
        axs[a].set_ylim((0,0.05))
        a += 1

        plt.tight_layout()
        return axs

    ############   TIME-STEPS: DESCRIPTIVE STATISTICS   ############

    def stats_per_t(self, scenarios=('ref',), funs=(np.mean,)):
        if type(scenarios) not in (list,tuple): scenarios = [scenarios]
        if type(funs) not in (list,tuple): funs = [funs]
        data = np.concatenate([np.array(self.d[s]) for s in scenarios], axis=0)
        return [fun(data, axis=0) for fun in funs]

    def rewards_per_episode(self, scenario='ref', resolution=25, outliers=0.2, acf_lags=100, show_dims=None):
        if show_dims is None:
            show_dims = (0, 5, 20, 100, int(0.5*self.T), int(0.9*self.T), int(0.95*self.T), int(0.99*self.T))

        axs = utils.Axes(6+2*(self.resolution==1), axsize=(8, 4), fontsize=14)

        axs[0].plot([np.mean([rs[i] for rs in self.d[scenario]]) for i in range(len(self.d[scenario][0]))])
        axs.labs(0, 'time', 'average reward\nover episodes')

        axs[1].plot([np.std([rs[i] for rs in self.d[scenario]]) for i in range(len(self.d[scenario][0]))])
        axs.labs(1, 'time', 'std(rewards)\nover episodes')

        ts = np.arange(0, len(self.d[scenario][0]), resolution)
        sns.boxplot(data=pd.DataFrame(dict(time=self.n_episodes(scenario) * list(ts),
                                           reward=[rs[i] for rs in self.d[scenario] for i in ts])),
                    x='time', y='reward', ax=axs[2], showmeans=True, fliersize=outliers)
        axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=90)

        sns.heatmap(self.G0[scenario], ax=axs[3],
                    cmap=sns.diverging_palette(h_neg=10, h_pos=130, sep=1, as_cmap=True), center=0)
        axs.labs(3, 't2', 't1', f'[{scenario:s}] rewards covariance matrix')

        for d in show_dims:
            axs[4].plot(self.G0[scenario][:,d], label=f't1={d:d}')
        axs.labs(4, 't2', 'Cov(r(t1), r(t2))')
        axs[4].legend(fontsize=10, loc='upper center')

        for d in show_dims:
            axs[5].plot(np.arange(d,self.T)-d//2, [self.G0[scenario][t, t+d] for t in range(self.T-d)],
                        label=f'dt={d:d}')
        axs.labs(5, 't', 'Cov(r(t), r(t+dt))')
        axs[5].legend(fontsize=10, loc='upper left')

        if self.resolution == 1:
            self.pacf(scenarios=(scenario,), partial=False, nlags=acf_lags, do_plot=True, ax=axs[6])
            self.pacf(scenarios=(scenario,), partial=True, nlags=acf_lags, do_plot=True, ax=axs[7])

        plt.tight_layout()

    def pacf(self, scenarios=('ref',), max_samples=None, time_range=None, merge_episodes=False,
             partial=True, remove_zero=True, alpha=0.01, nlags=50, normalize=False,
             do_plot=False, print_params=True, hline=0.03, ax=None, tit=None):
        fun = pacf if partial else acf
        if max_samples is None:
            max_samples = 1e5 if partial else 1e6
        if time_range is None:
            time_range = (0, self.T)
        n = time_range[1] - time_range[0]

        if normalize:
            M, S = self.stats_per_t(scenarios, (np.mean,np.std))
        else:
            M = np.zeros(len(self.d[scenarios[0]][0]))
            S = np.ones( len(self.d[scenarios[0]][0]))

        if merge_episodes:
            rewards = np.array([(ep[t]-M[t])/S[t] for s in scenarios for ep in self.d[s] for t in range(time_range[0],time_range[1]) ])
            if max_samples:
                rewards = rewards[:int(max_samples)]
            corr, err = fun(rewards, alpha=alpha, nlags=nlags)
            err = (err[:, 1] - err[:, 0]) / 2
        else:
            corr = np.zeros(1+nlags)
            corr2 = np.zeros(1+nlags)
            count = 0
            for s in scenarios:
                for ep in self.d[s]:
                    r = [(ep[t]-M[t])/S[t] for t in range(time_range[0],time_range[1])]
                    tmp = fun(r, alpha=None, nlags=nlags)
                    corr += tmp
                    corr2 += tmp**2
                    count += 1
                    if count*n >= max_samples:
                        break
                if count*n >= max_samples:
                    break
            corr /= count
            corr2 /= count
            err = np.sqrt((corr2 - corr**2) / count) * stats.norm.ppf(1-alpha)

        if remove_zero:
            corr, err = corr[1:], err[1:]

        if do_plot:
            ax = utils.Axes(1)[0] if ax is None else ax
            if hline is not None:
                ax.axhline(-hline, color='k', linestyle=':')
                ax.axhline( hline, color='k', linestyle=':')
            ax.errorbar(1 + np.arange(len(corr)), corr, yerr=err, fmt='.-', ecolor='r', capthick=2)
            if print_params:
                tmp = f'(#scenarios={len(scenarios):d} | samples={n*count:d} | time_range=({time_range[0]:d},{time_range[1]:d})\n'
                tmp += f'alpha={alpha:.3f} | thresh={hline if hline else "none"} | merged_episodes={merge_episodes})'
                tit = (tit+'\n'+tmp) if tit else tmp
            utils.labels(ax, 'Lag', 'PACF' if partial else 'ACF', tit, 14)
            if ax.get_ylim()[0] < -1: ax.set_ylim((-1,None))
            if ax.get_ylim()[1] >  1: ax.set_ylim((None,1))

        return corr, err, M, S

    def episode_variance_analysis(self, scenario='ref', resolution=1, eps=1e-9):
        T = self.T // resolution
        R = self.smooth_scenario_data(scenario, resolution)
        G0 = np.cov(R.transpose())
        G = np.zeros((T, T))
        for i in range(T):
            for j in range(i):
                # G0[i,j] = <G[i,:j+1],G[j,:j+1]> = sum(G[i,:j+1]*G[j,:j+1]) = sum(G[i,:j-1]*G[j,:j-1]) + Gij*Gjj
                G[i, j] = (G0[i, j] - np.sum(G[j, :j] * G[i, :j])) / G[j, j] if G[j, j]!=0 else 0
            tmp = G0[i, i] - np.sum(G[i, :i] ** 2)
            if tmp < -eps:
                raise ValueError()
            tmp = max(tmp, 0)
            G[i, i] = np.sqrt(tmp)

        Vt = np.array([G0[t,t] for t in range(len(G0))]) # time base
        V = np.sum(G, axis=0)**2 # orthogonal base

        if scenario not in self.episode_stats: self.episode_stats[scenario] = {}
        self.episode_stats[scenario][resolution] = dict(
            stds_t = np.sqrt(Vt),
            vars_t = Vt,
            vars_sum = np.sum(Vt),
            vars_perp = V,
            vars_gram = G,
            modeled_cum_var = [np.sum(np.sum(G[:t,:t],axis=0)**2) for t in range(len(G))],
            modeled_tot_var = np.sum(V),
            empiric_tot_var = (self.T * self.scores_std[scenario])**2,
            marginal_var_perp = V[-1] / Vt[-1],
        )

        return V, G

    def episode_variance_summary(self, scenario='ref', resolution=1, show_dims=None):
        T = self.T // resolution
        if show_dims is None:
            show_dims = (0, 1, 2, 5, 10, int(0.1*T), int(0.5*T), int(0.9*T), int(0.95*T), int(0.99*T))

        time_ticks = resolution * (1 + np.arange(T))
        R = self.smooth_scenario_data(scenario, resolution)
        G0 = np.cov(R.transpose())
        V, G = self.episode_variance_analysis(scenario=scenario, resolution=resolution)

        axs = utils.Axes(8, axsize=(8, 3.5), fontsize=14)
        a = 0

        sns.heatmap(G, ax=axs[a], cmap=sns.diverging_palette(h_neg=10, h_pos=130, sep=1, as_cmap=True), center=0)
        axs.labs(a, title=f'Gram matrix of covariance\n(sample resolution = {resolution:d} time-steps)')
        a += 1

        axs[a].plot(time_ticks, self.episode_stats[scenario][resolution]['vars_t'])
        axs.labs(a, 'Time step', 'Variance',
                 f'Naive (iid-based) total variance: {self.episode_stats[scenario][resolution]["vars_sum"]:.0f}\n' + \
                 f'Empirical total variance: {self.episode_stats[scenario][resolution]["empiric_tot_var"]:.0f}')
        a += 1

        axs[a].plot(time_ticks, [100*G[i,i]**2/G0[i,i] for i in range(len(G))])
        axs[a].set_ylim((0,100))
        axs.labs(a, 'Time step', 'Perpendicular variance\n[% of time-step]',
                 f'Marginal perpendicular variance = {100*self.episode_stats[scenario][resolution]["marginal_var_perp"]:.0f}%')
        a += 1

        axs[a].plot(time_ticks, 100 * np.array(self.episode_stats[scenario][resolution]['modeled_cum_var']) / \
                    self.episode_stats[scenario][resolution]['modeled_tot_var'])
        axs[a].set_ylim((0,100))
        axs.labs(a, 'Time step', 'Accumulated variance [%]')
        a += 1

        I = 100 * V / np.sum(V)
        axs[a].plot(time_ticks, I)
        axs.labs(a, 'Time step', 'New info [% of episode info]')
        a += 1

        axs[a].plot(time_ticks, np.cumsum(I))
        axs.labs(a, 'Time step', 'Accumulated info [%]',
                 f'Orthogonal-covs-based total variance: {self.episode_stats[scenario][resolution]["modeled_tot_var"]:.0f}\n' + \
                 f'Empirical total variance: {self.episode_stats[scenario][resolution]["empiric_tot_var"]:.0f}')
        a += 1

        for i in range(2):
            axs[a+i].axhline(0, color='k')
        for d in show_dims:
            axs[a+0].plot(time_ticks, G[:,d], label=f'dim={d:d} (sum={np.sum(G[:,d]):.1f})')
            axs.labs(a+0, 'Time step', 'STD component\nin dimension')
            axs[a+1].plot(time_ticks, np.cumsum(G[:,d]), label=f'dim={d:d}')
            axs.labs(a+1, 'Time step', 'Accum. STD component')
        axs[a+0].legend(fontsize=10, loc='center')
        axs[a+1].legend(fontsize=10, loc='center left')
        a += 2

        plt.tight_layout()

        return axs

    def reward_per_time_by_scenario(self, scenarios, resolution=1, s0='ref', outliers_size=2, fontsize=16, axs=None):
        if axs is None: axs = utils.Axes(2 + 3*(len(scenarios)>1))

        if len(scenarios) == 1:
            s = scenarios[0]
            ts = np.arange(0, len(self.d[s][0]), resolution)
            sns.boxplot(data=pd.DataFrame(dict(
                t=self.n_episodes(s) * list(self.resolution*(1+ts)),
                reward=[rs[i] for rs in self.d[s] for i in ts])),
                x='t', y='reward', ax=axs[0], showmeans=True, fliersize=outliers_size)
            axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
            utils.labels(axs[0], 't', 'reward', self.title, fontsize=fontsize)
            for s in scenarios:
                X = self.get_scenarios_data(s)
                axs[1].plot(self.resolution*np.arange(1,1+self.T), np.std(X, axis=0), label=s)
            utils.labels(axs[1], 't', 'std(reward)', self.title, fontsize=fontsize)

        else:
            X0 = self.d[s0]
            for s in scenarios:
                X = self.get_scenarios_data(s)
                axs[0].plot(self.resolution*np.arange(1,1+self.T), np.mean(X, axis=0), label=s)
                axs[1].plot(self.resolution*np.arange(1,1+self.T), np.std(X, axis=0), label=s)
                axs[2].plot(self.resolution*np.arange(1,1+self.T), np.mean(X, axis=0)-np.mean(X0, axis=0), label=s)
                axs[3].plot(self.resolution*np.arange(1,1+self.T), (np.mean(X,axis=0)-np.mean(X0,axis=0))/np.abs(np.mean(X0,axis=0)), label=s)
                axs[4].plot(self.resolution*np.arange(1,1+self.T), (np.mean(X,axis=0)-np.mean(X0,axis=0))/np.std(X0,axis=0), label=s)
            utils.labels(axs[0], 't', 'mean reward', self.title, fontsize=fontsize)
            utils.labels(axs[1], 't', 'std(reward)', self.title, fontsize=fontsize)
            utils.labels(axs[2], 't', 'mean reward - reference', self.title, fontsize=fontsize)
            utils.labels(axs[3], 't', '(reward-reference)/reference', self.title, fontsize=fontsize)
            utils.labels(axs[4], 't', 'standardized reward', self.title, fontsize=fontsize)
            for i in range(5):
                axs[i].legend(fontsize=10)

        return axs

    ############   TIME-STEPS: STATISTICAL TESTS   ############

    def get_scenario_data(self, s, n=None, n_episodes=None, n_additional=None, skip_episodes=0):
        if n_episodes is None:
            n_episodes = self.n_episodes(s) if n is None else n // self.T
        if n_additional is None:
            n_additional = 0 if n is None else n % self.T

        X = self.d[s][skip_episodes : skip_episodes+n_episodes].reshape(-1)
        if n_additional:
            X = np.concatenate((X, self.d[s][skip_episodes+n_episodes, :n_additional]))

        return X

    def show_cov_matrix(self, s0='ref', axs=None):
        if axs is None: axs = utils.Axes(9)

        STD = np.std(self.d[s0], axis=0)
        Sigma = np.cov(np.array(self.d[s0]).transpose())
        Corr = np.zeros_like(Sigma)
        invSigma = np.linalg.inv(Sigma)
        S = invSigma# + invSigma.transpose()
        n = S.shape[0]

        distance_from_diag = []
        Sigma_el = []
        Corr_el = []
        S_el = []
        for i in range(n):
            for j in range(n):
                distance_from_diag.append(self.resolution * (j-i))
                Sigma_el.append(Sigma[i,j])
                Corr[i,j] = Sigma[i,j]/(STD[i]*STD[j])
                Corr_el.append(Corr[i,j])
                S_el.append(S[i,j])

        dd = pd.DataFrame(dict(
            i = np.repeat(self.resolution*(1+np.arange(n)), n),
            j = n * list(self.resolution*(1+np.arange(n))),
            distance_from_diag = np.abs(np.array(distance_from_diag)),
            Cov_matrix_value = Sigma_el,
            Cov_matrix_abs_value = np.abs(np.array(Sigma_el)),
            Corr_value = Corr_el,
            Corr_abs_value = np.abs(np.array(Corr_el)),
            S_value = S_el,
            S_abs_value = np.abs(np.array(S_el)),
        ))

        sns.heatmap(Sigma, ax=axs[0], cmap=sns.diverging_palette(h_neg=10, h_pos=130, sep=1, as_cmap=True), center=0)
        utils.labels(axs[0], f't2 ($\\times {self.resolution}$)', f't1 ($\\times {self.resolution}$)', f'[{self.title:s}] Rewards covariance matrix', 14)
        sns.heatmap(S, ax=axs[1], cmap=sns.diverging_palette(h_neg=10, h_pos=130, sep=1, as_cmap=True), center=0)
        utils.labels(axs[1], f't2 ($\\times {self.resolution}$)', f't1 ($\\times {self.resolution}$)', f'[{self.title:s}] Inverted covariance matrix', 14)
        sns.boxplot(data=dd, x='distance_from_diag', y='Cov_matrix_value', ax=axs[2], showmeans=True)
        utils.labels(axs[2], 'distance from diagonal ($|i-j|$)', 'covariance ($\Sigma_{ij}$)', fontsize=16)
        sns.boxplot(data=dd, x='distance_from_diag', y='S_value', ax=axs[3], showmeans=True)
        sns.boxplot(data=dd, x='distance_from_diag', y='Cov_matrix_abs_value', ax=axs[4], showmeans=True)
        sns.boxplot(data=dd, x='distance_from_diag', y='S_abs_value', ax=axs[5], showmeans=True)
        sns.heatmap(Corr, ax=axs[6], cmap=sns.diverging_palette(h_neg=10, h_pos=130, sep=1, as_cmap=True), center=0)
        utils.labels(axs[6], f't2 ($\\times {self.resolution}$)', f't1 ($\\times {self.resolution}$)', f'[{self.title:s}] Rewards correlations', 14)
        sns.boxplot(data=dd, x='distance_from_diag', y='Corr_value', ax=axs[7], showmeans=True)
        utils.labels(axs[7], 'distance from diagonal ($|i-j|$)', 'correlation(i,j)', fontsize=16)
        sns.boxplot(data=dd, x='distance_from_diag', y='Corr_abs_value', ax=axs[8], showmeans=True)
        utils.labels(axs[8], 'distance from diagonal ($|i-j|$)', '|correlation(i,j)|', fontsize=16)
        for i in list(range(2,6))+[7,8]:
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90, fontsize=8)
            axs[i].set_title(self.title, fontsize=16)

        return dd, axs

    def show_testers_weights(self, tests, ax=None, s0='ref', normalize=True, logscale=False, resolution=1,
                             fontsize=16, **kwargs):
        if ax is None: ax = utils.Axes(1)[0]
        w = {}
        for nm, constructor_test_args in tests.items():
            tester = constructor_test_args[0](X=self.d[s0], title=nm, **constructor_test_args[1])
            try:
                w[nm] = tester.get_temporal_weights()
                if normalize:
                    w[nm] /= np.sum(w[nm])
                if logscale:
                    w[nm] = np.abs(w[nm])
                ax.plot(self.resolution*resolution*np.arange(1,1+tester.T), w[nm], label=nm, **kwargs)
            except NotImplementedError:
                pass
                # warn(f'Tester {nm:s} has no explicit weights.')
        if logscale:
            ax.set_yscale('log')
        ax.legend(fontsize=12)
        title = self.title
        if resolution>1:
            title += f'(sample resolution = {resolution:d})'
        utils.labels(ax, 't', 'weight', title, fontsize=fontsize)
        return w, ax

    def run_tests(self, tests, meta_scenario, ns, s0='ref', sides=(None,), resolution=1, ref_testers=tuple(),
                  mark_sign=None, verbose=0, to_save=False, filename='tmp'):
        # configuration
        # if side is None: side = self.side
        if mark_sign is None: mark_sign = (isinstance(sides,dict) or len(sides)>1)
        scenarios = [s for s in self.scenarios if s.startswith(meta_scenario)]
        if verbose >= 1:
            print('Tested scenario:', meta_scenario)

        # initialization
        tot_sides = int(np.sum([len(ss) for ss in sides.values()])) \
            if isinstance(sides,dict) else (len(sides) * len(tests))
        n_rows = tot_sides * len(ns) * len(scenarios)
        res = pd.DataFrame(dict(
            test = n_rows * [''],
            side = n_rows * [0],
            scenario = n_rows * [''],
            n = n_rows * [0],
            s = n_rows * [np.nan],
            p = n_rows * [np.nan],
            z = n_rows * [np.nan]
        ))
        def update_res(count, nm, sd, sc, n, stat, p, z):
            res.iloc[count, 0] = f'{nm:s}{"+" if sd>0 else ("-" if sd<0 else "0")}' if mark_sign else nm
            res.iloc[count, 1] = sd
            res.iloc[count, 2] = sc
            res.iloc[count, 3] = n
            res.iloc[count, 4] = stat
            res.iloc[count, 5] = p
            res.iloc[count, 6] = z

        # run tests
        t0 = time()
        testers = []
        count = 0
        for nm, constructor_test_args in tests.items():
            # get/build tester
            tester = None
            for tmp in ref_testers:
                if tmp.title == nm:
                    tester = tmp # dictionary instead of list would make more sense...
            if tester is None:
                args = utils.update_dict(constructor_test_args[1],
                                         dict(X=self.d[s0], seed=self.seed, resolution=resolution, B=1000, title=nm),
                                         force=False, copy=True)
                tester = constructor_test_args[0](**args)
            testers.append(tester)
            t_test = len(constructor_test_args)>2 and constructor_test_args[2]
            curr_sides = sides[tester.title] if isinstance(sides,dict) else sides
            # run test
            for n in ns:
                for s in scenarios:
                    X = self.get_scenario_data(s, n//self.resolution, skip_episodes=0)
                    for side in curr_sides:
                        if t_test:
                            p, z = tester.t_test(X, side=side)
                            stat = np.nan
                        else:
                            p, z, stat = tester.bootstrap_main(X, side=side, return_statistic=True)
                        update_res(count, nm, side, s, n, stat, p, z)
                        count += 1

            tester.clean_gpu(level=2)

            if verbose >= 1:
                print(f'\t{nm:s} done.\t({time()-t0:.0f} [s])')

        if to_save:
            with open(f'outputs/non_sequential/{filename:s}.pkl', 'wb') as fd:
                pkl.dump((res, testers), fd)

        return res, testers

    def run_pairs_tests(self, tests, meta_scenarios, ns, s0='ref', n_blocks=100, resolution=1, self_compare=True, ref_testers=tuple(),
                  verbose=0, to_save=False, filename='tmp'):
        # initialization
        pairs = [(s1,s2) for s1 in meta_scenarios for s2 in meta_scenarios if (self_compare or s1!=s2)]
        n_rows = len(pairs) * len(ns) * len(tests) * n_blocks
        res = pd.DataFrame(dict(
            scenario1 = n_rows * [''],
            scenario2 = n_rows * [''],
            n = n_rows * [0],
            test = n_rows * [''],
            block = n_rows * [0],
            s = n_rows * [np.nan],
            p = n_rows * [np.nan],
            z = n_rows * [np.nan]
        ))
        def update_res(count, s1, s2, n, test, b, stat, p, z):
            res.iloc[count, 0] = s1
            res.iloc[count, 1] = s2
            res.iloc[count, 2] = n
            res.iloc[count, 3] = test
            res.iloc[count, 4] = b
            res.iloc[count, 5] = stat
            res.iloc[count, 6] = p
            res.iloc[count, 7] = z

        # run tests
        t0 = time()
        testers = []
        count = 0
        for nm, constructor_test_args in tests.items():
            # get/build tester
            tester = None
            for tmp in ref_testers:
                if tmp.title == nm:
                    tester = tmp # dictionary instead of list would make more sense...
            if tester is None:
                args = utils.update_dict(constructor_test_args[1],
                                         dict(X=self.d[s0], seed=self.seed, resolution=resolution, B=1000, title=nm),
                                         force=False, copy=True)
                tester = constructor_test_args[0](**args)
            testers.append(tester)

            # run test
            for pair in pairs:
                for n in ns:
                    for b in range(n_blocks):
                        X1 = self.get_scenario_data(f'{pair[0]:s}_{b:02d}', n//self.resolution)
                        X2 = self.get_scenario_data(f'{pair[1]:s}_{b:02d}', n//self.resolution)
                        p, z, stat = tester.bootstrap2_main(X1, X2, return_statistic=True)
                        update_res(count, pair[0], pair[1], n, nm, b, stat, p, z)
                        count += 1

            tester.clean_gpu(level=2)

            if verbose >= 1:
                print(f'\t{nm:s} done.\t({time()-t0:.0f} [s])')

        if to_save:
            with open(f'outputs/non_sequential/{filename:s}.pkl', 'wb') as fd:
                pkl.dump((res, testers), fd)

        return res, testers

    def run_tests_with_skips(self, tests, s, s0='ref', ns=None, skips=None, side=None, resolution=1,
                             max_repetitions=1000, verbose=0, to_save=False, filename='tmp'):
        # configuration
        if side is None: side = self.side
        if ns is None: ns = [self.d[s].shape[0] * self.T]
        # if skips is None: skips = list(range(0, self.d[s].shape[0]-np.max(ns)//self.T, int(np.ceil(np.max(ns)/self.T))))
        if skips is None: skips = {n: list(range(0, int(min(self.d[s].shape[0]-n//self.T, max_repetitions*np.ceil(n/self.T))),
                                                 int(np.ceil(n/self.T)))) for n in ns}
        if verbose >= 1:
            print('Tested scenario:', s)

        # initialization
        if isinstance(skips, dict):
            n_rows = len(tests) * np.sum(list(map(len, list(skips.values()))))
        else:
            n_rows = len(tests) * len(ns) * len(skips)
        res = pd.DataFrame(dict(
            test = n_rows * [''],
            n0 = n_rows * [0],
            n = n_rows * [0],
            s = n_rows * [np.nan],
            p = n_rows * [np.nan],
            z = n_rows * [np.nan]
        ))
        def update_res(count, nm, n0, n, stat, p, z):
            res.iloc[count, 0] = nm
            res.iloc[count, 1] = n0
            res.iloc[count, 2] = n
            res.iloc[count, 3] = stat
            res.iloc[count, 4] = p
            res.iloc[count, 5] = z

        # run tests
        t0 = time()
        testers = []
        count = 0
        for nm, constructor_test_args in tests.items():
            args = utils.update_dict(constructor_test_args[1],
                                     dict(X=self.d[s0], seed=self.seed, resolution=resolution, B=1000, title=nm),
                                     force=False, copy=True)
            tester = constructor_test_args[0](**args)
            testers.append(tester)
            t_test = constructor_test_args[2] if len(constructor_test_args)>2 else False
            for n in ns:
                curr_skips = skips[n] if isinstance(skips, dict) else skips
                for skip in curr_skips:
                    n0 = skip * self.T
                    X = self.get_scenario_data(s, n, skip_episodes=skip)
                    if t_test:
                        p, z = tester.t_test(X, side=side)
                        stat = np.nan
                    else:
                        p, z, stat = tester.bootstrap_main(X, side=side, return_statistic=True)
                    update_res(count, nm, n0, n, stat, p, z)
                    count += 1

            tester.clean_gpu(level=2)

            if verbose >= 1:
                print(f'{nm:s} done.\t({time()-t0:.0f} [s])')

        if to_save:
            with open(f'outputs/non_sequential/{filename:s}.pkl', 'wb') as fd:
                pkl.dump((res, testers), fd)

        return res, testers

    def analyze_tests(self, res, testers, alphas=(0.01,0.05),
                      qqplots=False, pval_vs_n=False, z_values=True, stat_vs_bootstrap=False, errors_summary=True,
                      ns_to_plot=None, tests_to_plot=6, unite_sides=False,
                      log_n_axis=True, power_confidence_interval=95, axs=None):
        ns = np.unique(res.n)
        if ns_to_plot is None: ns_to_plot = ns
        if axs is None:
            n_axs = qqplots*min(len(ns),len(ns_to_plot)) + pval_vs_n*1*len(testers) + \
                    stat_vs_bootstrap*1 + z_values*1 + errors_summary*len(alphas)
            axs = utils.Axes(n_axs, axsize=(6.5,4))

        a = 0
        if qqplots:
            for n in ns_to_plot:
                dd = res[res.n == n]
                ps = []
                labs = []
                for test in np.unique(dd.test):
                    ps.append(dd[dd.test == test].p)
                    lab = test + ' ('
                    for alpha in alphas:
                        lab += f"{(ps[-1] < alpha).mean():.3f}, "
                    lab = lab[:-2] + ')'
                    labs.append(lab)

                Analyzer.qqplot_unif(ps, a=0, b=1, xlab='Scenario', ylab='P-value', labs=labs, ax=axs[a])
                axs[a].legend(loc='upper left', fontsize=10)
                tit = f'n_samples = {n:d}, n_repetitions = {len(dd)//len(testers):d}\n('
                for alpha in alphas:
                    tit += f"P(p<{alpha:.2f}), "
                tit = tit[:-2] + ')'
                axs.labs(a, title=tit)
                a += 1

        if pval_vs_n:
            for test in sorted([tester.title for tester in testers]):
                dd = res[res.test == test]
                n0s = [aa[0] for aa in Counter(dd.n0).most_common(tests_to_plot)]
                dd = dd[dd.n0.isin(n0s)]
                dd['first_episode'] = [f'ep={x//self.T:d}' for x in dd.n0]
                # sns.lineplot(data=dd, hue='first_episode', x='n', y='p', marker='o', ax=axs[a+1])
                sns.lineplot(data=dd, hue='first_episode', x='n', y='z', marker='o', ax=axs[a])
                if log_n_axis:
                    # axs[a+1].set_xscale('log')
                    axs[a].set_xscale('log')
                # axs[a+1].legend(fontsize=10)
                axs[a].legend(fontsize=10)
                # axs[a+1].set_ylim((0,1))
                # utils.labels(axs[a+1], title=test)
                utils.labels(axs[a], title=test)
                a += 1

        if stat_vs_bootstrap:
            centralization = {tester.title:{n:0 for n in ns_to_plot} for tester in testers}
            dd = pd.DataFrame()
            for tester in testers:
                nn = len(ns_to_plot)
                # plot bootstrap distribution
                tmp = pd.DataFrame(dict(
                    tester = nn*tester.B*[tester.title],
                    n_samples = np.repeat(ns_to_plot, tester.B),
                    centralized_statistic = np.concatenate([tester.boot[n//self.resolution//tester.resolution] for n in ns_to_plot])
                ))
                for i, n in enumerate(ns_to_plot):
                    centralization[tester.title][n] = tmp[tmp.n_samples == n].centralized_statistic.mean()
                    tmp.loc[i*tester.B:(i+1)*tester.B, 'centralized_statistic'] -= centralization[tester.title][n]
                dd = pd.concat((dd,tmp))
            sns.boxplot(data=dd, x='n_samples', hue='tester', y='centralized_statistic', showfliers=False, ax=axs[a])
            axs[a].legend(fontsize=10)
            # plot tests statistics
            for i, tester in enumerate(testers):
                xx = []
                yy = []
                for j, n in enumerate(ns_to_plot):
                    tmp = res[(res.test==tester.title) & (res.n==n)]
                    xx.extend(len(tmp) * [j+(i+0.5-len(testers)/2)/(len(testers)+0.5)])
                    yy.extend((tmp.s-centralization[tester.title][n]).values)
                axs[a].plot(xx, yy, 'ro', markersize=1)
            # axs[a].set_xticklabels(axs[a].get_xticklabels(), rotation=20)
            a += 1

        if z_values:
            rr = res.copy()
            if unite_sides:
                rr['test'] = [(tt[:-1] if tt[-1] in ('+','-') else tt) for tt in rr.test]
            sns.boxplot(data=rr[rr.n.isin(ns_to_plot)], x='n', hue='test', y='z', showmeans=True, ax=axs[a])
            axs[a].legend(fontsize=10)
            utils.labels(axs[a], 'n_samples', 'z-value', fontsize=16)
            a += 1

        if errors_summary:
            P = {}
            for alpha in alphas:
                axs[a].axhline(alpha, color='k')
                if power_confidence_interval:
                    res['reject'] = res.p < alpha
                    sns.lineplot(data=res, x='n', hue='test', y='reject', marker='o',
                                 ci=power_confidence_interval, ax=axs[a])
                else:
                    P[alpha] = pd.DataFrame()
                    for test in [tester.title for tester in testers]:
                        dd = res[res.test == test]
                        tmp = dd.groupby(['n0' if 'n0' in dd.columns else 'scenario', 'n']).apply(lambda d: d.p < alpha)
                        tmp = pd.DataFrame(tmp.groupby('n').apply(np.mean))
                        tmp['n_samples'] = tmp.index
                        tmp.reset_index(drop=True, inplace=True)
                        tmp['test'] = test
                        P[alpha] = pd.concat((P[alpha], tmp))
                    P[alpha].reset_index(drop=True, inplace=True)
                    sns.lineplot(data=P[alpha], hue='test', x='n_samples', y='p', marker='o', ax=axs[a])

                axs[a].legend(fontsize=10)
                n_skips = [len(res[res.n==n]) // len(testers) for n in ns]
                if np.min(n_skips) == np.max(n_skips):
                    n_skips = f'{n_skips[0]:d}'
                else:
                    n_skips = f'{np.min(n_skips):d}-{np.max(n_skips):d}'
                utils.labels(axs[a],
                             f'number of samples\n(tests per sample-size: {n_skips:s})',
                             f'P(p-value < {alpha:.2f})',
                             fontsize=15)
                axs[a].set_ylim((0, None))
                if log_n_axis: axs[a].set_xscale('log')
                a += 1

        plt.tight_layout()
        return axs

    def weighted_vs_simple(self, s, ax, T1=None, T2=None, s0='ref', resolution=10, B=1000, fontsize=16,
                           ns=(10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)):
        # T1 = Stats.SimpleMean(X=self.d[s0], B=B, resolution=resolution, title='Simple')
        if T1 is None: T1 = Stats.WeightedCUSUM(X=self.d[s0], B=B, resolution=resolution, K=0, title='CUSUM')
        if T2 is None: T2 = Stats.WeightedMean(X=self.d[s0], B=B, resolution=resolution, title='Weighted')
        dd = pd.DataFrame(dict(
            test=len(ns) * (B * [T1.title] + B * [T2.title]),
            n=[m for l in [2 * B * [n] for n in ns] for m in l],
            test_statistic=B * len(ns) * 2 * [0],
        ))
        ss = []
        ws = []
        count = 0
        for n in ns:
            X = self.get_scenario_data(s, n)
            T1.bootstrap_main(X, side=self.side)
            ss.append(T1.calc(T1.downsample(X)))
            dd.loc[count:count + B - 1, 'test_statistic'] = T1.boot[n // T1.resolution]
            count += B
            T2.bootstrap_main(X, side=self.side)
            ws.append(T2.calc(T2.downsample(X)))
            dd.loc[count:count + B - 1, 'test_statistic'] = T2.boot[n // T2.resolution]
            count += B
        sns.boxplot(data=dd, x='n', y='test_statistic', hue='test', showmeans=False, ax=ax)
        ax.plot(np.arange(len(ss)) - 0.2, ss, 'bo', markersize=4)
        ax.plot(np.arange(len(ws)) + 0.2, ws, 'ro', markersize=4)
        ax.set_title(s, fontsize=fontsize)

    ############   SEQUENTIAL TESTS   ############

    def rolling_pvals(self, testers, X, t0=None, horizons=(1,10,100), side=-1, stop_alphas=None,
                      downsampled=False, detailed=False, print_freq=0):
        # setup
        if side is None: side = self.side
        horizons = sorted(horizons)
        n_hors = len(horizons)
        hor_max = max(horizons)
        if t0 is None: t0 = hor_max * self.T
        frequency = testers[0].T
        sample_unit = testers[0].resolution
        n_samples = len(X) * (sample_unit if downsampled else 1)
        n_episodes = int( (n_samples - t0) // self.T )
        n_hor_tests = n_episodes * frequency
        n_testers = len(testers)

        # initialization
        pvals = {tester.title:[] for tester in testers}
        t_end = {tester.title:-1 for tester in testers}
        rejecting_horizon = {tester.title:-1 for tester in testers}
        dd = None
        if detailed:
            dd = pd.DataFrame(dict(
                t = np.repeat(self.resolution*np.arange(0, n_episodes * self.T, sample_unit), n_testers * n_hors),
                test = n_hor_tests * list(np.repeat([tester.title for tester in testers], n_hors)),
                lookback = n_testers * n_hor_tests * horizons,
                p = n_testers * n_hors * n_hor_tests * [np.nan],
                z = n_testers * n_hors * n_hor_tests * [np.nan],
            ))

        # calculate p-vals
        P = []
        Z = []
        end_count = n_testers
        start_time = time()
        for t in np.arange(0, n_episodes * self.T, sample_unit):
            if end_count == 0:
                break

            for tester in testers:
                if t_end[tester.title] >= 0:
                    if detailed:
                        P.append(np.nan)
                        Z.append(np.nan)
                    continue
                for hor in horizons:
                    tf = t0 + t
                    ti = int(self.T * np.ceil(tf/self.T-hor))
                    x = X[ti//sample_unit:tf//sample_unit] if downsampled else X[ti:tf]
                    p, z = tester.bootstrap_main(x, downsampled=downsampled, side=side, rolling=(hor, sample_unit))
                    pvals[tester.title].append(p)
                    if detailed:
                        P.append(p)
                        Z.append(z)
                    if stop_alphas is not None:
                        if isinstance(stop_alphas, dict):
                            if tester.title in stop_alphas and p < stop_alphas[tester.title]:
                                end_count -= 1
                                t_end[tester.title] = self.resolution * t
                                rejecting_horizon[tester.title] = hor
                        elif p < stop_alphas:
                                end_count -= 1
                                t_end[tester.title] = self.resolution * t
                                rejecting_horizon[tester.title] = hor

            if print_freq and (t % (sample_unit * print_freq)) == 0:
                print(f'\t{t:04d}/{n_episodes*self.T:04d}:\tz={z:.2f}\t({time()-start_time:.0f} [s])')

        for tester in testers:
            tester.reset_rolling_statistic()

        if dd is not None:
            dd['p'] = P
            dd['z'] = Z
            dd = dd[dd.p.notna()]

        return pvals, dd, t_end, rejecting_horizon

    def rolling_test(self, scenario_prefix=None, scenarios=None, s0='ref', horizons=(5,50), frequency=10, alpha=0.01,
                     skip_episodes=0, max_episodes=None, prefix_scenarios_prefix=None, random_prefix=False, seed=None,
                     testers=None, tester_const=None, tester_args=None, load_boots=False,
                     save_res=False, detailed=False, **kwargs):
        # setup
        if max_episodes is None:
            max_episodes = max(horizons)
        if scenarios is None:
            scenarios = [s for s in self.scenarios if s!=s0 and s.startswith(scenario_prefix)]
        if testers is None:
            if tester_const is None: tester_const = Stats.WeightedMean
            if tester_args is None: tester_args = {}
            tester_args = utils.update_dict(tester_args, dict(X=self.d[s0],B=3000,title='W-Mean'), force=False)
            tester_args = utils.update_dict(tester_args, dict(resolution=self.T//frequency), force=True)
            testers = [tester_const(**tester_args)]
            if load_boots:
                testers[0].load_bootstrap(n_boots=len(horizons)*frequency)

        res = {}
        all_dd = pd.DataFrame() if detailed else None
        for s in scenarios:
            # get prefix data
            if seed is not None: np.random.seed(seed)
            if prefix_scenarios_prefix:
                s_prefix = np.random.choice([scenario for scenario in self.scenarios if scenario.startswith(prefix_scenarios_prefix)])
            else:
                s_prefix = s0
            X0 = self.d[s_prefix]
            if random_prefix:
                ids0 = np.random.choice(np.arange(X0.shape[0]), max(horizons), replace=True)
                X0 = X0.take(ids0, axis=0)
            else:
                X0 = X0[-max(horizons):, :]
            X0 = X0.reshape(-1)

            # get test data
            X1 = self.d[s]
            if max_episodes: X1 = X1[:max_episodes, :]
            if skip_episodes: X1 = X1[skip_episodes:, :]
            X1 = X1.reshape(-1)

            # rolling test
            X = np.concatenate((X0,X1))
            pvals, dd, t_reject, hor_reject = self.rolling_pvals(testers, X, horizons=horizons, stop_alphas=alpha,
                                                                 detailed=detailed, **kwargs)
            if detailed:
                dd['scenario'] = s
                all_dd = pd.concat((all_dd, dd))
            res[s] = (pvals, t_reject, hor_reject)

        if save_res:
            fname = Analyzer.pvals_bootstrap_filename(horizons, frequency,
                                                      title=save_res if isinstance(save_res,str) else testers[0].title)
            with open(fname, 'wb') as fd:
                pkl.dump((res, all_dd), fd)

        return res, all_dd

    def bootstrap_rolling_pvals(self, tester, horizons=(5,50), h_forward=None, seed=None, detailed=False, **kwargs):
        if seed is not None: np.random.seed(seed)
        horizons = sorted(horizons)
        hor_max = max(horizons)
        if h_forward is None: h_forward = hor_max
        X0 = np.array(tester.X0)
        ids = np.random.choice(np.arange(X0.shape[0]), hor_max+h_forward, replace=True)
        X = X0.take(ids, axis=0).reshape(-1)
        return self.rolling_pvals([tester], X, horizons=horizons, downsampled=True, detailed=detailed, **kwargs)

    @staticmethod
    def pvals_bootstrap_filename(horizons=(1,10,100), frequency=100, B=None, title='W-Mean'):
        fname = f'outputs/pvals_{title:s}_'
        if B is not None:
            fname += f'bootstrap{B:d}_'
        fname += f'freq{frequency:d}_hors'
        for h in horizons:
            fname += f'{h:d}.'
        fname = fname[:-1]
        fname += '.pkl'
        return fname

    def alpha_budget_bootstrap(self, alpha0=0.05, horizons=(5,50), h_forward=None, frequency=10, B=1000, s0='ref',
                               tester=None, tester_const=None, tester_args=None, load_boots=False, save_boots=False,
                               save_res=False, print_freq=0, distributed=0, detailed=False, **kwargs):
        # setup
        if h_forward is None: h_forward = max(horizons)
        if tester is None:
            if tester_const is None: tester_const = Stats.WeightedMean
            if tester_args is None: tester_args = {}
            tester_args = utils.update_dict(tester_args, dict(X=self.d[s0],B=3000,title='W-Mean'), force=False)
            tester_args = utils.update_dict(tester_args, dict(resolution=self.T//frequency), force=True)
            tester = tester_const(**tester_args)
        if load_boots:
            tester.load_bootstrap(n_boots=len(horizons)*frequency)
        dd = None

        # bootstrap
        if distributed:
            raise NotImplementedError()
            from concurrent.futures import ProcessPoolExecutor
            bootstrap_rolling_pvals_wrapper = lambda b: \
                self.bootstrap_rolling_pvals(tester=tester, horizons=horizons, seed=b,
                                             detailed=False, print_freq=0)[0]
            tpool = ProcessPoolExecutor(max_workers=distributed)
            pvals = tpool.map(bootstrap_rolling_pvals_wrapper, range(B))
            pvals = np.array(list(pvals))

        else:
            if detailed:
                dd = pd.DataFrame()
            pvals = np.ones(B)
            t0 = time()
            for b in range(B):
                p, d, _, _ = self.bootstrap_rolling_pvals(tester, horizons, h_forward, seed=b, detailed=detailed, **kwargs)
                pvals[b] = np.min(p[tester.title])
                if detailed:
                    d['b'] = b
                    dd = pd.concat((dd,d))
                if print_freq and (b % print_freq) == 0:
                    print(f'\t{b+1:03d}/{B:d}\t({time()-t0:.0f} [s])')
            if save_boots:
                tester.save_bootstrap()

        if save_res:
            fname = Analyzer.pvals_bootstrap_filename(horizons, frequency, B, tester.title)
            with open(fname, 'wb') as fd:
                pkl.dump((pvals,dd), fd)

        # find alpha
        alpha = np.quantile(pvals, alpha0)

        return alpha, pvals, dd

    def analyze_sequential_results(self, dd, alphas=None, alpha0=None, max_hor=None, h_forward=None, log_pval=True,
                                   pvals_dists=True, rej_vs_time=True, horizons_of_rej=True,
                                   power_confidence_interval=95, axs=None):
        # dd cols: ('scenario_group', 'scenario', 'test', 'min_pval', 'rej', 't_rej', 'horizon_rej')
        #    (scenario_group is assumed to be homogeneous in data frame)
        if h_forward is None: h_forward = max(max_hor) if max_hor is not None else dd.horizon_rej.max()
        tests = dd.test.values[:len(np.unique(dd.test))]
        horizons = np.unique(dd.horizon_rej)
        n_repetitions = len(dd) // len(tests)

        if axs is None:
            axs = utils.Axes(3)
        a = 0

        if pvals_dists:
            if alphas is not None:
                axs[a].axhline(1 if alpha0 is None else alpha0, color='k', linestyle='--')
            for test in tests:
                min_pvals = dd[dd.test == test].min_pval
                if alphas is not None:
                    min_pvals /= alphas[test]
                    if alpha0 is not None:
                        min_pvals *= alpha0
                utils.plot_quantiles(min_pvals, ax=axs[a], label=test)
            utils.labels(axs[a], 'quantile [%]', 'normalized min p-value\nover sequential tests',
                         f'[{self.title:s}] {dd.scenario_group.values[0]:s}', 14)
            if log_pval: axs[a].set_yscale('log')
            axs[a].legend(fontsize=10)
            a += 1

        if rej_vs_time:
            Tmax = h_forward * self.T*self.resolution
            self.plot_episode_transitions(axs[a], 1) # 1->Tmax for all transitions
            ts = np.arange(0, Tmax + 0.1, Tmax / 100)
            if power_confidence_interval:
                rejs = pd.DataFrame()
                for test in tests:
                    d = dd[dd.test == test]
                    t_rejs = d.t_rej
                    total_rejs = (t_rejs>=0).mean()
                    lab = f'{test:s} ({100 * total_rejs:.0f}%)'
                    for t in ts: # make more efficient?
                        tmp = pd.DataFrame(dict(t=len(d)*[t], test=len(d)*[lab]))
                        n_rejs = int(((t_rejs>=0) & (t_rejs<=t)).sum())
                        tmp['reject'] = n_rejs*[100] + (len(d)-n_rejs)*[0]
                        rejs = pd.concat((rejs, tmp))
                sns.lineplot(data=rejs, x='t', hue='test', y='reject', linewidth=2.5,
                             ci=power_confidence_interval, ax=axs[a])
            else:
                for test in tests:
                    t_rejs = dd[dd.test == test].t_rej
                    total_rejs = (t_rejs>=0).mean()
                    rejs = []
                    for t in ts:
                        rejs.append(100 * ((t_rejs>=0) & (t_rejs<=t)).mean())
                    axs[a].plot(ts, rejs, label=test+f' ({100*total_rejs:.0f}%)')
            axs[a].legend(fontsize=12)
            utils.labels(axs[a], 't', 'rejected [%]',
                         f'[{self.title:s}] {dd.scenario_group.values[0]:s}', 17)
            axs[a].set_xscale('log')
            a += 1

        if horizons_of_rej:
            tmp = pd.DataFrame({
                'test' : np.repeat(tests, len(horizons)),
                'lookback_horizon' : len(tests) * list(horizons),
                'rejections [%]' : len(tests) * len(horizons) * [0.0],
            })
            count = 0
            for test in tests:
                d = dd[dd.test == test]
                for h in horizons:
                    tmp.loc[count, 'rejections [%]'] = 100 * (d.horizon_rej==h).sum() / n_repetitions
                    count += 1
            sns.barplot(ax=axs[a], data=tmp, x='lookback_horizon', hue='test', y='rejections [%]')
            utils.labels(axs[a], 'lookback horizon responsible for rejection\n(-1 = not rejected)',
                         'rejections [%]', f'[{self.title:s}] {dd.scenario_group.values[0]:s}', 16)
            axs[a].legend(fontsize=10)
            a += 1

        return axs

    ############   SUMMARY   ############

    def run_summarizing_tests(self, tests=None, scenarios=None, s0='ref', s_pre='H0', resolution=1, default_tester_args=None,
                              ns=(100, 300, 1000, 3000, 10000, 30000), non_seq_sides=(None,), save_name='tmp', load_non_seq=False, load_alpha_tuning=False, verbose=2,
                              alpha0=0.05, lookback_horizons=(5,50), seq_test_len=None, seq_B=1000, seq_detailed=False, levels=3):
        if seq_test_len is None:
            seq_test_len = max(lookback_horizons)
        if default_tester_args is None:
            default_tester_args = dict(B=3000)
        if tests is None:
            tests = dict(
                simple_mean = (Stats.SimpleMean, dict()),
                std_mean = (Stats.IndependentWeightedMean, dict(squared_weights=False)),
                CUSUM_mean = (Stats.CUSUM, dict(weights=True)),
                invS_mean = (Stats.WeightedMean, dict()),
            )
        for t in tests:
            utils.update_dict(tests[t][1], default_tester_args)
        if scenarios is None:
            scenarios = list(np.unique([s[:-3] for s in self.scenarios if s!=s0]))
            if verbose >= 2:
                print('Scenarios:', scenarios)

        time0 = time()

        # non-sequential tests
        if load_non_seq:
            with open(f'outputs/non_sequential/all_scenarios_tests_{save_name:s}.pkl', 'rb') as fd:
                res_non_seq, testers, tests, _ = pkl.load(fd)
            if verbose >= 1:
                print(f'Non-sequential tests loaded.\t({time() - time0:.0f} [s])')
        else:
            testers = []
            res_non_seq = pd.DataFrame()
            for s in scenarios:
                res, testers = self.run_tests(tests, s, s0=s0, ns=ns, resolution=resolution, ref_testers=testers,
                                              sides=non_seq_sides, verbose=verbose-1)
                res['scenario'] = s
                res_non_seq = pd.concat((res_non_seq, res))

            if save_name:
                data_to_save = (res_non_seq, testers, tests, dict(resolution=resolution))
                self.res[save_name] = data_to_save
                with open(f'outputs/non_sequential/all_scenarios_tests_{save_name:s}.pkl', 'wb') as fd:
                    pkl.dump(data_to_save, fd)

            if verbose >= 1:
                print(f'Non-sequential tests done.\t({time()-time0:.0f} [s])')

        if levels <= 1:
            return None, res_non_seq, testers, tests

        # sequential tests tuning
        if load_alpha_tuning:
            if not isinstance(load_alpha_tuning, str):
                load_alpha_tuning = save_name
            with open(f'outputs/sequential/alpha_tuning_{load_alpha_tuning:s}.pkl', 'rb') as fd:
                all_dd, all_pvals, alphas, seq_args = pkl.load(fd)
            alpha0 = seq_args['alpha0']
            if verbose >= 1:
                print(f'Sequential tests tuning loaded.\t({time() - time0:.0f} [s])')
        else:
            alphas = {}
            all_pvals = {}
            all_dd = pd.DataFrame()
            for tester in testers:
                alpha, pvals, dd = self.alpha_budget_bootstrap(alpha0=alpha0, horizons=lookback_horizons, h_forward=seq_test_len,
                                                               frequency=self.T//resolution, B=seq_B, s0=s0,
                                                               tester=tester, save_res=False, detailed=True,
                                                               print_freq=100*(verbose>=2))
                alphas[tester.title] = alpha
                all_pvals[tester.title] = pvals
                dd['test'] = tester.title
                all_dd = pd.concat((all_dd, dd))

                if verbose >= 2:
                    print(f'\tsequential {tester.title:s} tuned.\t({time() - time0:.0f} [s])')

            if save_name:
                with open(f'outputs/sequential/alpha_tuning_{save_name:s}.pkl', 'wb') as fd:
                    pkl.dump((all_dd, all_pvals, alphas, dict(alpha0=alpha0, h_forward=seq_test_len)), fd)

            if verbose >= 1:
                print(f'Sequential tests tuned.\t({time() - time0:.0f} [s])')

        if levels <= 2:
            return None, res_non_seq, testers, tests

        # sequential tests
        all_res = {}
        all_dd = pd.DataFrame() if seq_detailed else None
        for s in scenarios:
            res, dd = self.rolling_test(scenario_prefix=s, s0=s0, horizons=lookback_horizons, max_episodes=seq_test_len, frequency=self.T//resolution,
                                        alpha=alphas, prefix_scenarios_prefix=s_pre, random_prefix=False, testers=testers,
                                        save_res=save_name, detailed=seq_detailed)
            all_res[s] = res
            if seq_detailed:
                all_dd = pd.concat((all_dd, dd))
            if verbose >= 2:
                print(f'\tsequential {s:s} tested.\t({time() - time0:.0f} [s])')
        # all_res[s_pre][s][0(pval)/1(t_rej)/2(hor_rej)][test_name]
        n_rows = np.sum([len(all_res[s]) for s in all_res]) * len(testers)
        res_seq = pd.DataFrame(index=np.arange(n_rows), columns=('scenario_group','scenario','test','min_pval','rej','t_rej','horizon_rej'))
        count = 0
        for s in scenarios:
            for ss in all_res[s]:
                for tester in testers:
                    res_seq.loc[count, 'scenario_group'] = s
                    res_seq.loc[count, 'scenario'] = ss
                    res_seq.loc[count, 'test'] = tester.title
                    res_seq.loc[count, 'min_pval'] = np.min(all_res[s][ss][0][tester.title])
                    res_seq.loc[count, 'rej'] = all_res[s][ss][1][tester.title] >= 0
                    res_seq.loc[count, 't_rej'] = all_res[s][ss][1][tester.title]
                    res_seq.loc[count, 'horizon_rej'] = all_res[s][ss][2][tester.title]
                    count += 1

        if save_name:
            with open(f'outputs/sequential/tests_{save_name:s}.pkl', 'wb') as fd:
                pkl.dump((res_seq, all_dd), fd)

        if verbose >= 1:
            print(f'Sequential tests done.\t({time() - time0:.0f} [s])')

        return res_seq, res_non_seq, testers, tests

    def load_testers(self, fname, verbose=0):
        with open(f'outputs/non_sequential/all_scenarios_tests_{fname:s}.pkl', 'rb') as fd:
            all_res, testers, tests, args = pkl.load(fd)
        if verbose >= 1:
            print(tests)
        return testers

    def analysis_summary(self, load_file, s0='ref', scenarios=None, ns=None, do_general=True,
                         do_sequential=True, do_non_seq=True, axs_args=None, verbose=1):
        time0 = time()

        # load data
        with open(f'outputs/non_sequential/all_scenarios_tests_{load_file:s}.pkl', 'rb') as fd:
            all_res, testers, tests, args = pkl.load(fd)
        if scenarios is None: scenarios = list(np.unique(all_res.scenario))
        if ns is None: ns = sorted(list(np.unique(all_res.n)))
        resolution = args['resolution']

        if verbose >= 1:
            print(f'Data loaded.\t({time()-time0:.0f} [s])')

        # initialize axes
        if axs_args is None: axs_args = dict(W=3, axsize=(5.7,3.9))
        axs = utils.Axes(do_general*(2+7+9+2+1) + \
                         do_non_seq*(3*len(scenarios)+3*int(np.ceil(len(ns)/3))) + \
                         do_sequential*(2 + 1 + 3*len(scenarios)),
        **axs_args)
        a = 0

        if do_general:
            # scenarios scores
            dd = self.m.copy()
            dd['scenario'] = [s[:-3] for s in dd.scenario]
            dd = dd[dd.scenario.isin(scenarios)]
            dd.sort_values('scenario', inplace=True)
            axs[a + 0].axhline(dd[dd.scenario == 'H0'].score.mean(), color='k', linestyle='--')
            sns.boxplot(data=dd, x='scenario', y='score', showmeans=True, ax=axs[a + 0], showfliers=False)
            axs[a + 0].set_xticklabels(axs[a + 0].get_xticklabels(), rotation=30)
            utils.labels(axs[a + 0], 'scenario', 'score',
                         f'[{self.title:s}] scores distribution over episodes', fontsize=14)
            for s in scenarios:
                scores0 = self.m[self.m.scenario==s0].score
                scores = dd[dd.scenario==s].score
                z = (scores.mean()-scores0.mean()) / np.sqrt(scores.var()/len(scores) + scores0.var()/len(scores0))
                lab = f'{s:s} (z={z:.1f})'
                utils.plot_quantiles(dd[dd.scenario==s].score, axs[a+1], plot_mean=True, label=lab)
            utils.labels(axs[a+1], 'episode quantile [%]', 'score', self.title, 15)
            axs[a+1].legend(fontsize=10)
            a += 2

            # rewards vs. time
            self.reward_per_time_by_scenario((s0,), s0=s0, axs=(axs[a + 0], axs[a + 1]),
                                             resolution=int(np.ceil(self.T/20)))
            axs.labs(a+0, title=f'[{self.title}] Reference episodes', fontsize=15)
            axs.labs(a+1, title=f'[{self.title}] Reference episodes', fontsize=15)
            self.reward_per_time_by_scenario(scenarios, s0=s0, axs=[axs[a+2+i] for i in range(5)])
            a += 7

            if verbose >= 1:
                print(f'EDA done.\t({time()-time0:.0f} [s])')

            # testers weights
            self.show_cov_matrix(s0, [axs[a+i] for i in range(9)])
            a += 9
            self.show_testers_weights(tests, axs[a+0], s0=s0, resolution=resolution)
            self.show_testers_weights(tests, axs[a+1], s0=s0, resolution=resolution, logscale=True)
            a += 2

            a += 1

        if do_non_seq:
            # non-sequential tests comparison
            for s in scenarios:
                res = all_res[all_res.scenario==s]
                self.analyze_tests(res, testers, stat_vs_bootstrap=False, axs=[axs[a+i] for i in range(4)])
                for i in range(3):
                    axs[a+i].set_title(f'[{self.title:s}] {s:s}', fontsize=15)
                a += 3
            for n in ns:
                rr = all_res.copy()
                # rr['test'] = [(tt[:-1] if tt[-1] in ('+', '-') else tt) for tt in rr.test]
                sns.boxplot(data=rr[rr.n == n], ax=axs[a], x='scenario', hue='test', y='z', showmeans=True)
                axs[a].legend(fontsize=10)
                axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=12, rotation=30)
                axs.labs(a, 'scenario', 'z-value', f'[{self.title:s}] n_samples = {int(n):d}', fontsize=15)
                a += 1
            if len(ns) % 3 != 0:
                a += 3 - (len(ns) % 3)

            if verbose >= 1:
                print(f'Non-sequential tests comparison done.\t({time()-time0:.0f} [s])')

        if do_sequential:
            # sequential tests: alpha tuning
            with open(f'outputs/sequential/alpha_tuning_{load_file:s}.pkl', 'rb') as fd:
                all_dd, all_pvals, alphas, seq_args = pkl.load(fd)
            alpha0 = seq_args['alpha0']
            h_forward = seq_args['h_forward']
            max_hor = int(all_dd.lookback.max())
            axs[a].axvline(100*alpha0, color='k', linestyle='--')
            for test in all_pvals:
                utils.plot_quantiles(all_pvals[test], axs[a], label=test)
            axs.labs(a, f'cumulative-alpha per {h_forward:d} episodes', 'alpha per test', self.title, fontsize=14)
            axs[a].legend(fontsize=10)
            axs[a].set_yscale('log')
            a += 1
            sns.boxplot(data=all_dd, x='lookback', hue='test', y='p', ax=axs[a])
            axs.labs(a, 'lookback horizon', 'p-value', self.title, fontsize=14)
            axs[a].legend(fontsize=10)
            a += 1

            a += 1
            # add figure: from all_dd: for each tester: for few seeds: axhline(zval(alphas[tester.title])); plot(x=t, y=z-val, by=horizon)

            # sequential tests
            with open(f'outputs/sequential/tests_{load_file:s}.pkl', 'rb') as fd:
                res_seq, all_dd = pkl.load(fd)
                # res_seq: scenario_group, scenario, test, min_pval, rej, t_rej, horizon_rej
            for s in scenarios:
                self.analyze_sequential_results(res_seq[res_seq.scenario_group==s], alphas=alphas, alpha0=alpha0,
                                                max_hor=max_hor, h_forward=h_forward, axs=[axs[a+i] for i in range(3)])
                a += 3

            if verbose >= 1:
                print(f'Sequential tests comparison done.\t({time()-time0:.0f} [s])')

        plt.tight_layout()
        return axs
