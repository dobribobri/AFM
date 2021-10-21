# -*- coding: utf-8 -*-
import os
from typing import Union, Iterable, List, Tuple
# from typing import Callable
import numpy as np
import networkx as nx
# from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import dill
import re
import yaml
import itertools
from scipy.special import softmax
from multiprocessing import Process, Manager
from collections import defaultdict
import sys

from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 14})


class _signal:
    def __init__(self, mode: str = 'gaussian_derivative', **kwargs):
        self.mode = mode

        _c = np.power(10., -12)
        self.s = 0.5 * _c
        _T = 50 * _c
        self.t0 = _T / 3.
        self.H0 = 0.25
        self.omega0 = 2 * np.pi * 150 * np.power(10, 9)

        self.const = 1.

        self.__dict__.update(kwargs)

    def set(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        if self.mode == 'gaussian':
            _p = (t - self.t0) / self.s
            return self.H0 * np.exp(-_p * _p) * np.sin(self.omega0 * t)
        if self.mode == 'gaussian_derivative':
            _p = (t - self.t0) / self.s
            _e = np.exp(-_p * _p)
            _s = self.s * self.s
            return self.H0 * self.omega0 * _e * np.cos(self.omega0 * t) - \
                   (2 * self.H0 * (t - self.t0) * _e * np.sin(self.omega0 * t)) / _s
        if self.mode == 'zero':
            return 0 * t
        if self.mode == 'const':
            return self.const * t
        return None

    def __str__(self):
        return str(self.__dict__)


class _Signal:
    def __init__(self, n_nodes: int, *signals):
        self.n_nodes = n_nodes
        self.signals = []
        if signals:
            self.set(signals)

    def set(self, signals) -> None:
        assert self.n_nodes <= len(signals), 'wrong shape'
        self.signals = [s for s in signals]

    def append(self, signal: _signal) -> None:
        # assert self.n_nodes > len(self.signals), 'overflow'
        self.signals.append(signal)

    def extend(self, signals: Union[List[_signal], Iterable[_signal]]) -> None:
        for s in signals:
            self.append(s)

    def __call__(self, n: int, t: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
        if n >= self.n_nodes:
            raise IndexError('n >= n_nodes')
        return self.signals[n].__call__(t)

    def __str__(self):
        s = ''
        for _i, signal in enumerate(self.signals):
            s += '{}| {}\n'.format(_i, signal)
        return s

    @staticmethod
    def generate(n_nodes: int,
                 _input: Union[List[float], Iterable[float]] = None,
                 _input_mode='gaussian_derivative') -> '_Signal':
        if _input is None:
            _input = []
        assert n_nodes >= len(_input), 'wrong shape'
        H = _Signal(n_nodes=n_nodes)
        H.extend([_signal(mode=_input_mode, H0=H0, t0=0.) for H0 in _input])
        H.extend([_signal(mode='zero') for _ in range(n_nodes - len(_input))])
        return H


class AFM:
    def __init__(self,
                 n_nodes: int = 10,
                 dt: float = 100 * np.power(10., -12) / 1000, T: float = 100 * np.power(10., -12),
                 adj_mat: Union[np.ndarray, Iterable] = None,
                 omega_e: float = 2. * np.pi * 1.75 * np.power(10., 9),
                 omega_ex: float = 2 * np.pi * 27.5 * np.power(10., 12),
                 alpha: float = 0.01,
                 sigma: float = 2. * np.pi * 4.32,
                 j_dc: float = 0.0,
                 gamma: float = 2. * np.pi * 28 * np.power(10., 9),
                 H: _Signal = _Signal(10, *[_signal(
                     'gaussian_derivative',
                     s=0.5 * np.power(10., -12),
                     t0=50 * np.power(10., -12) / 3.,
                     H0=0.25,
                     omega0=2. * np.pi * 150 * np.power(10., 9)
                 )] * 10),
                 phi_ini: Union[np.ndarray, Iterable] = None,
                 dot_phi_ini: Union[np.ndarray, Iterable] = None):

        if n_nodes is None and adj_mat is None:
            raise ValueError("n_nodes or adj_mat must be specified")

        self.n_nodes = n_nodes
        if n_nodes is None:
            self.n_nodes = len(adj_mat)

        self.dt, self.T = dt, T

        self.adj_mat = None
        self.set_adj_mat(adj_mat)

        self.alpha = alpha
        self.omega_e, self.omega_ex = omega_e, omega_ex
        self.sigma, self.j_dc = sigma, j_dc
        self.gamma = gamma
        self.H = H

        self.phi = None
        self.set_phi(phi_ini)

        self.dot_phi = None
        self.set_dot_phi(dot_phi_ini)

        self.time_elapsed = 0.

    def save(self, name: str) -> None:
        with open('{}.afm'.format(name), 'wb') as dump:
            dill.dump(self, dump, recurse=True)

    @staticmethod
    def load(name: str) -> 'AFM':
        with open('{}.afm'.format(name), 'rb') as dump:
            return dill.load(dump)

    def save_txt(self, fname: str) -> None:
        with open(fname, 'w') as file:
            file.write('\n\nn_nodes\n')
            file.write('{}'.format(self.n_nodes))
            file.write('\n\ndt\n')
            file.write('{}'.format(self.dt))
            file.write('\n\nT\n')
            file.write('{}'.format(self.T))
            file.write('\n\nadj_mat\n')
            for line in self.adj_mat:
                s = ''
                for item in line:
                    s += '{}, '.format(item)
                file.write('{}\n'.format(s))
            file.write('\n\nomega_e\n')
            file.write('{}'.format(self.omega_e))
            file.write('\n\nomega_ex\n')
            file.write('{}'.format(self.omega_ex))
            file.write('\n\nalpha\n')
            file.write('{}'.format(self.alpha))
            file.write('\n\nsigma\n')
            file.write('{}'.format(self.sigma))
            file.write('\n\nj_dc\n')
            file.write('{}'.format(self.j_dc))
            file.write('\n\ngamma\n')
            file.write('{}'.format(self.gamma))
            file.write('\n\nphi\n')
            s = ''
            for phi in self.phi:
                s += '{}, '.format(phi)
            file.write('{}'.format(s))
            file.write('\n\ndot_phi\n')
            s = ''
            for dot_phi in self.dot_phi:
                s += '{}, '.format(dot_phi)
            file.write('{}'.format(s))
            file.write('\n\nSignals\n')
            file.write('{}'.format(self.H))

    @staticmethod
    def load_txt(fname: str) -> 'AFM':
        obj = AFM()
        with open(fname, 'r') as file:
            try:
                for line in file:
                    name = re.sub(r'[^A-Za-z_0-9]', '', re.sub(r'[\r\n\t ]', '', line))
                    if name in ['n_nodes']:
                        val = int(re.sub(r'[^0-9.]', '', re.sub(r'[\r\n\t ]', '', file.readline())))
                        setattr(obj, name, val)
                    if name in ['omega_e', 'omega_ex', 'alpha', 'sigma', 'j_dc', 'gamma', 'dt', 'T']:
                        val = float(re.sub(r'[^0-9.e+-]', '', re.sub(r'[\r\n\t ]', '', file.readline())))
                        setattr(obj, name, val)
                    if name == 'adj_mat':
                        adj_mat = []
                        for _ in range(obj.n_nodes):
                            row = re.split(',', re.sub(r'[^0-9.,]', '', re.sub(r'[\r\n\t ]', '', file.readline())))
                            row = [float(item) for item in row if item]
                            adj_mat.append(row[:obj.n_nodes])
                        setattr(obj, name, np.array(adj_mat))
                    if name in ['phi', 'phi_ini', 'phi_init']:
                        row = re.split(',', re.sub(r'[^0-9.,]', '', re.sub(r'[\r\n\t ]', '', file.readline())))
                        row = [float(item) for item in row if item]
                        setattr(obj, 'phi', row[:obj.n_nodes])
                    if name in ['dot_phi', 'dot_phi_ini', 'dot_phi_init']:
                        row = re.split(',', re.sub(r'[^0-9.,]', '', re.sub(r'[\r\n\t ]', '', file.readline())))
                        row = [float(item) for item in row if item]
                        setattr(obj, 'dot_phi', row[:obj.n_nodes])
                    if name in ['Signals', 'signals']:
                        H = _Signal(obj.n_nodes)
                        for _ in range(obj.n_nodes):
                            row = re.split(r'\|',
                                           re.sub(r'[^A-Za-z_0-9.,\'\-{}|:]', '',
                                                  re.sub(r'[\r\n\t ]', '', file.readline())))
                            # print(row)
                            d = yaml.safe_load(row[1])
                            signal = _signal()
                            for key in d.keys():
                                if key not in ['mode']:
                                    d[key] = float(d[key])
                                setattr(signal, key, d[key])
                            H.append(signal)
                        setattr(obj, 'H', H)
            except ValueError:
                print('bad values')
                exit(-1)
            except IndexError:
                print('bad file')
                exit(-1)
        return obj

    def set_adj_mat(self, adj_mat: Union[np.ndarray, Iterable] = None) -> None:
        if adj_mat is None:
            self.adj_mat = nx.to_numpy_array(nx.erdos_renyi_graph(n=self.n_nodes, p=1))
        else:
            assert self.n_nodes == len(adj_mat), 'wrong shape'
            # assert np.asarray(adj_mat == adj_mat.T).all(), 'adj_mat must be symmetric'
            self.adj_mat = np.asarray(adj_mat)

    def set_phi(self, phi_ini: Union[np.ndarray, Iterable] = None) -> None:
        if phi_ini is None:
            self.init_zero_phi()
        else:
            assert self.n_nodes == len(phi_ini), 'wrong shape'
            self.phi = np.asarray(phi_ini)

    def init_random_phi(self) -> None:
        self.phi = 2 * np.pi * np.random.random(size=self.n_nodes)

    def init_zero_phi(self) -> None:
        self.phi = np.array([0 for _ in range(self.n_nodes)])

    def set_dot_phi(self, dot_phi_ini: Union[np.ndarray, Iterable] = None) -> None:
        if dot_phi_ini is None:
            self.init_zero_dot_phi()
        else:
            assert self.n_nodes == len(dot_phi_ini), 'wrong shape'
            self.dot_phi = np.asarray(dot_phi_ini)

    def init_random_dot_phi(self) -> None:
        self.dot_phi = 2 * np.pi * np.random.random(size=self.n_nodes)

    def init_zero_dot_phi(self) -> None:
        self.dot_phi = np.array([0 for _ in range(self.n_nodes)])

    def init_random(self) -> None:
        self.init_random_phi()
        self.init_random_dot_phi()

    def init_zero(self) -> None:
        self.init_zero_phi()
        self.init_zero_dot_phi()

    @staticmethod
    def __check_rearrange_mode(mode: str) -> None:
        if mode not in ['pdpd', 'dpdp', 'ppdd', 'ddpp', 'pppp', 'dddd']:
            raise ValueError("allowed modes: \'pdpd\', \'dpdp\', \'ppdd\', \'ddpp\', \'pppp\', \'dddd\'")

    def get_state(self, mode: str = 'pdpd') -> np.ndarray:
        self.__check_rearrange_mode(mode)
        p, d = np.reshape(self.phi, (-1, 1)), \
               np.reshape(self.dot_phi, (-1, 1))
        out = np.concatenate((p, d), axis=-1)  # 'pdpd'
        if mode == 'pppp':
            out = p
        if mode == 'dddd':
            out = d
        if mode == 'ppdd':
            out = np.concatenate((p, d), axis=0)
        if mode == 'ddpp':
            out = np.concatenate((d, p), axis=0)
        if mode == 'dpdp':
            out = np.concatenate((d, p), axis=-1)
        return out.reshape(1, -1)[0]

    # def __d(self, y, t) -> list:      # for scipy.integrate.odeint
    def __d(self, t, y) -> list:  # for scipy.integrate.solve_ivp
        _phi, _theta = y[::2], y[1::2]
        d = []
        for _i, values in enumerate(zip(_phi, _theta)):
            phi, theta = values
            d.extend([
                theta,
                -self.alpha * self.omega_ex * theta +
                self.omega_ex * np.sum(self.adj_mat[_i] * _theta) -
                0.5 * self.omega_e * self.omega_ex * np.sin(2 * phi) +
                self.omega_ex * self.sigma * self.j_dc +
                self.gamma * self.H(_i, t)
            ])
        return d

    def integrate(self, _dt: float = None, _t_stop: float = None, return_mode: Union[str, None] = 'ppdd',
                  change_state: bool = False, method='RK45') -> Union[np.ndarray, None]:
        dt, t_stop = self.dt, self.T
        if _dt:
            dt = _dt
        if _t_stop:
            t_stop = _t_stop

        t = np.linspace(0, t_stop, int(t_stop / dt), endpoint=True) + self.time_elapsed
        initial_state = self.get_state('pdpd')
        # series = odeint(self.__d, initial_state, t)
        sol = solve_ivp(self.__d, y0=initial_state,
                        t_span=(self.time_elapsed, t_stop + self.time_elapsed),
                        t_eval=t, method=method)
        series = sol.y.T

        p, d = series[:, ::2], series[:, 1::2]
        if change_state:
            self.phi, self.dot_phi = p[-1, :], d[-1, :]
        if return_mode:
            self.__check_rearrange_mode(return_mode)
            if return_mode == 'pppp':
                return p
            if return_mode == 'dddd':
                return d
            out = np.zeros_like(series)
            if return_mode == 'pdpd':
                out[:, :] = series[:, :]
            if return_mode == 'dpdp':
                out[:, ::2], out[:, 1::2] = d, p
            if return_mode == 'ppdd':
                out[:, :self.n_nodes], out[:, self.n_nodes:] = p, d
            if return_mode == 'ddpp':
                out[:, :self.n_nodes], out[:, self.n_nodes:] = d, p
            return out
        return

    def step(self, _dt: float = None, n: int = 2, method='RK45', _return_dot_phi=False) -> Union[None, np.ndarray]:
        dt = self.dt
        if _dt:
            dt = _dt

        self.integrate(_dt=dt / n, _t_stop=dt, change_state=True, return_mode=None, method=method)
        self.time_elapsed += dt
        if _return_dot_phi:
            return self.dot_phi
        return

    def get_phi(self) -> np.ndarray:
        return self.phi

    def get_dot_phi(self) -> np.ndarray:
        return self.dot_phi

    def execute(self, v='0.1.1'):
        if v == '0.1.1':
            ts = self.integrate(return_mode='ppdd', method='RK45')
            _, n = ts.shape
            fig, ax = plt.subplots(nrows=2, figsize=(12, 10))
            y_labels = [r'$\varphi$', r'$\dot{\varphi}$']
            l_labels = [r'$\varphi_' + str(j) + '$' for j in range(n // 2)]
            l_labels.extend([r'$\dot{\varphi_' + str(j) + '}$' for j in range(n // 2)])
            k = 0
            for i in range(2):
                ax[i].set_ylabel(y_labels[i], rotation=0, fontsize=20, labelpad=20)
                ax[i].set_xlabel('time, s')
                for j in range(n // 2):
                    ax[i].plot(np.linspace(0, self.T, len(ts)), ts[:, k], label=l_labels[k])
                    k += 1
                ax[i].legend(loc='best', frameon=False)
            plt.savefig('figure.png', dpi=300)
            plt.show()


class MLP(AFM):
    def __init__(self,
                 input_layer_size: int, output_layer_size: int,
                 hidden_layer_sizes=(4,),
                 oriented: bool = False,
                 coupling: float = np.power(10., -3),
                 dt: float = 100 * np.power(10., -12) / 1000,
                 T: float = 100 * np.power(10., -12) / 1000 * 60.,
                 omega_e: float = 2. * np.pi * 1.75 * np.power(10., 9),
                 omega_ex: float = 2 * np.pi * 27.5 * np.power(10., 12),
                 alpha: float = 0.01,
                 sigma: float = 2. * np.pi * 4.32,
                 j_dc: float = 0.0,
                 gamma: float = 2. * np.pi * 28 * np.power(10., 9)):

        if hidden_layer_sizes is None:
            hidden_layer_sizes = []
        n_nodes = input_layer_size + int(np.sum(hidden_layer_sizes)) + output_layer_size
        sizes = [input_layer_size] + [ls for ls in hidden_layer_sizes if ls] + [output_layer_size]
        G = MLP.multilayered_graph(oriented, *sizes)
        adj_mat = nx.to_numpy_array(G) * coupling
        adj_mat *= np.random.random(adj_mat.shape)

        H = _Signal.generate(n_nodes)  # zero

        super(MLP, self).__init__(n_nodes=n_nodes, dt=dt, T=T, adj_mat=adj_mat,
                                  omega_e=omega_e, omega_ex=omega_ex, alpha=alpha, sigma=sigma,
                                  j_dc=j_dc, gamma=gamma, H=H)
        self.init_zero()

        self.input_layer_size, self.output_layer_size = input_layer_size, output_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes

        self.genes = list(itertools.product(range(self.n_nodes), range(self.n_nodes)))
        self.coupling = coupling

    @staticmethod
    def multilayered_graph(oriented=False, *subset_sizes) -> nx.Graph:
        extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
        layers = [range(start, end) for start, end in extents]
        if oriented:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for _i, layer in enumerate(layers):
            G.add_nodes_from(layer, layer=_i)
        for layer1, layer2 in nx.utils.pairwise(layers):
            G.add_edges_from(itertools.product(layer1, layer2))
        return G

    def forward(self, x: np.ndarray, _t_stop: float = None,
                _normalize=False, _softmax=False, _max_pooling=False) -> np.ndarray:
        assert self.input_layer_size == len(x), 'wrong shape (x)'
        t_stop = self.T
        if _t_stop:
            t_stop = _t_stop
        self.H = _Signal.generate(self.n_nodes, _input=x, _input_mode='gaussian_derivative')
        dot_phi = self.integrate(_t_stop=t_stop, return_mode='dddd', change_state=False)[:, -self.output_layer_size:]
        # plt.plot(dot_phi)
        peaks = np.max(np.abs(dot_phi), axis=0)
        if _normalize or _softmax:
            peaks /= np.max(peaks)
        if _softmax:
            peaks = softmax(peaks)
        if _max_pooling:
            index = np.argmax(peaks)
            peaks = np.zeros_like(peaks)
            peaks[index] = 1
        return peaks

    def forward_multiple(self, X: np.ndarray, _t_stop: float = None,
                         _normalize=False, _softmax=False, _max_pooling=False,
                         verbose=False) -> np.ndarray:
        out = []
        for k, x in enumerate(X):
            out.append(self.forward(x, _t_stop, _normalize, _softmax, _max_pooling))
            if verbose:
                print('\r{:.2f}% done'.format((k + 1) / len(X) * 100), end='', flush=True)
        if verbose:
            print('.\n')
        return np.asarray(out)

    def error_on_batch(self, X: np.ndarray, y: np.ndarray, _t_stop: float = None,
                       _normalize=False, _softmax=False,
                       verbose=False) -> float:
        rows_x, _ = X.shape
        rows_y, columns_y = y.shape
        assert rows_x == rows_y, 'X and y must have the same number of samples'
        assert self.output_layer_size == columns_y, 'wrong shape (y)'
        out = self.forward_multiple(X, _t_stop, _normalize, _softmax, verbose)
        return np.linalg.norm(y - out)

    def crossover(self, obj: 'MLP', percentage: float = 0.5) -> None:
        sh, sw = self.adj_mat.shape
        oh, ow = obj.adj_mat.shape
        assert sh == sw == oh == ow == self.n_nodes, 'wrong shape'
        transfer = [obj.genes[i] for i in
                    np.random.choice(range(len(obj.genes)), int(len(obj.genes) * percentage))]
        for i, j in transfer:
            self.adj_mat[i][j] = obj.adj_mat[i][j]

    def mutation(self, n: int):
        mutable = [self.genes[i] for i in np.random.choice(range(len(self.genes)), n)]
        r = np.random.random(n) * 2 - 1
        for k, pos in enumerate(mutable):
            i, j = pos
            a = self.adj_mat[i][j]
            val = a + r[k] * (a + self.coupling * 0.01)
            if val <= 0.:
                val = 0.
            self.adj_mat[i][j] = val


if __name__ == '__main__':

    # model = AFM(n_nodes=1)
    # model.save('test1')
    # model.save_txt('test1.txt')

    # model = AFM.load('test1')
    # model = AFM.load_txt('test1.txt')
    # model.save_txt('test2.txt')

    # model = AFM(n_nodes=10)
    # model.save_txt('test3.txt')
    # model = AFM.load_txt('test3.txt')

    # Test 1
    # model = AFM.load_txt('test1.txt')
    # T = 100 * np.power(10., -12)
    # ts = model.integrate(dt=T/1000, t_stop=T, return_mode='pdpd')
    # _, n = ts.shape
    #
    # fig, ax = plt.subplots(nrows=n, figsize=(12, 10))
    # y_labels = [r'$\varphi$', r'$\dot{\varphi}$']
    # line_styles = ['-', '-']
    # for i in range(n):
    #     ax[i].set_ylabel(y_labels[i], rotation=0, fontsize=20, labelpad=20)
    #     ax[i].set_xlabel('time, ps')
    #     ax[i].plot(np.linspace(0, 100, len(ts)), ts[:, i],
    #                label=y_labels[i], color='black', ls=line_styles[i])
    #     ax[i].legend(loc='best', frameon=False)
    # plt.show()

    # Test 2
    # plt.ion()
    # fig, ax = plt.subplots(nrows=2, figsize=(12, 10))
    # y_labels = [r'$\varphi$', r'$\dot{\varphi}$']
    # for i in range(2):
    #     ax[i].set_xlim((0, 100))
    #     ax[i].set_ylabel(y_labels[i], rotation=0, fontsize=20, labelpad=20)
    #     ax[i].set_xlabel('time, ps')
    # ax[0].set_ylim((-0.003, 0.003))
    # ax[1].set_ylim((-1 * np.power(10., 10), 0.5 * np.power(10., 10)))
    #
    # T = 100 * np.power(10., -12)
    # N = 1000
    # dt = T / N
    # tx, phi, dot_phi = [], [], []
    # for i in range(N):
    #     model.step(dt=dt)
    #
    #     tx.append(model.time_elapsed * np.power(10., 12))
    #
    #     phi.append(model.phi[0])
    #     ax[0].plot(tx, phi, color='black', ls='-')
    #
    #     dot_phi.append(model.dot_phi[0])
    #     ax[1].plot(tx, dot_phi, color='black', ls='-')
    #
    #     plt.show(block=False)
    #     # plt.savefig('pic/{}.png'.format(i), dpi=300)
    #     fig.canvas.flush_events()
    # plt.ioff()
    # plt.show()

    # Test 3
    # signal = _Signal(n_nodes=2)
    # signal.extend([_signal(H0=0.25), _signal(H0=0.5)])
    #
    # model = AFM(n_nodes=2,
    #             adj_mat=np.array([[0., 7. * np.power(10., -4)],
    #                               [2. * np.power(10., -4), 0.]]),
    #             H=signal)
    #
    # model.save_txt('model.txt')

    # # EXE v0.0.1
    # model = AFM.load_txt('model.txt')
    #
    # # T = 100 * np.power(10., -12)
    # # ts = model.integrate(_dt=T / 1000, _t_stop=T, return_mode='ppdd', method='RK45')
    #
    # ts = model.integrate(return_mode='ppdd', method='RK45')
    #
    # # N = 1000
    # # ts = np.zeros((N, 4))
    # # tx = []
    # # for i in range(N):
    # #     model.step(_dt=T / N)
    # #     tx.append(model.time_elapsed * np.power(10., 12))
    # #     phi = model.get_phi()
    # #     dot_phi = model.get_dot_phi()
    # #     ts[i, 0] = phi[0]
    # #     ts[i, 1] = phi[1]
    # #     ts[i, 2] = dot_phi[0]
    # #     ts[i, 3] = dot_phi[1]
    #
    # _, n = ts.shape
    # # print(n)
    #
    # fig, ax = plt.subplots(nrows=2, figsize=(12, 10))
    # y_labels = [r'$\varphi$', r'$\dot{\varphi}$']
    # l_labels = [r'$\varphi_' + str(j) + '$' for j in range(n // 2)]
    # l_labels.extend([r'$\dot{\varphi_' + str(j) + '}$' for j in range(n // 2)])
    # k = 0
    # for i in range(2):
    #     ax[i].set_ylabel(y_labels[i], rotation=0, fontsize=20, labelpad=20)
    #     ax[i].set_xlabel('time, ps')
    #     for j in range(n // 2):
    #         ax[i].plot(np.linspace(0, 100, len(ts)), ts[:, k], label=l_labels[k])
    #         k += 1
    #     ax[i].legend(loc='best', frameon=False)
    #
    # plt.savefig('figure.png', dpi=300)
    # plt.show()

    from sklearn.datasets import load_iris
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    X, y = load_iris(return_X_y=True)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    y_categorical = np.zeros((len(y), len(le.classes_)))
    for i, _y in enumerate(y):
        y_categorical[i][_y] = 1
    y = y_categorical
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    n_workers = 16
    n_individuals = 500
    # n_lonely = 5  # TBD
    n_best = 5
    n_new = 50
    n_genes_mutable = 10
    n_epochs = sys.maxsize

    initial_population = []
    for i in range(n_individuals):
        initial_population.append(MLP(input_layer_size=4, output_layer_size=3, hidden_layer_sizes=None))

    population = initial_population

    if not os.path.exists('best'):
        os.makedirs('best')

    plt.ion()
    fig, ax = plt.subplots(nrows=2, figsize=(10, 7))
    # ax[0].set_title('loss')
    # ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('value')
    # ax[1].set_title('f1-score')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel(r'$\%$')

    losses = defaultdict(list)
    f1_train_list, f1_test_list = [], []
    colors = ['crimson', 'forestgreen', 'black']

    for epoch in range(n_epochs):

        with Manager() as manager:
            scores = manager.list()
            processes = []
            for i, individual in enumerate(population):
                p = Process(target=lambda _scores, _i, _individual:
                _scores.append((_i, _individual.error_on_batch(X_train, y_train, _softmax=True))),
                            args=(scores, i, individual,)
                            )
                processes.append(p)
            for i in range(0, len(processes), n_workers):
                for j in range(i, i + n_workers):
                    if j < len(processes):
                        processes[j].start()
                for j in range(i, i + n_workers):
                    if j < len(processes):
                        processes[j].join()
            scores = list(scores)

        errors = sorted(scores, key=lambda item: item[1])
        values = [val for _, val in errors]

        s = '\r{:.2f}%\t-\tepoch: {}\t-\tloss: {:.2f} (1 best),\t{:.2f} ({} best),\t{:.2f} (total)\t-\t'.format(
            (epoch + 1.) / n_epochs * 100,
            epoch + 1,
            np.mean(values[:1]),
            np.mean(values[:n_best]),
            n_best,
            np.mean(values)
        )
        losses['best'].append(np.mean(values[:1]))
        losses['{} best'.format(n_best)].append(np.mean(values[:n_best]))
        losses['total'].append(np.mean(values))
        epochs = np.array(list(range(0, epoch + 1))) + 1
        for i, key in enumerate(losses.keys()):
            ax[0].plot(epochs, losses[key], label='loss: {}'.format(key), color=colors[i])
        if not epoch:
            ax[0].legend(loc='best', frameon=False)

        best, other = [i for i, _ in errors[:n_best]], [i for i, _ in errors[n_best:]]
        p = np.asarray(population, dtype=object)
        best_individuals, other_individuals = p[best], p[other]

        best_individuals[0].save_txt('best/{}.txt'.format(epoch))
        best_out_train = best_individuals[0].forward_multiple(X_train, _softmax=True, _max_pooling=True)
        best_out_test = best_individuals[0].forward_multiple(X_test, _softmax=True, _max_pooling=True)
        f1_train = f1_score(y_train, best_out_train, average='weighted')
        f1_test = f1_score(y_test, best_out_test, average='weighted')
        s += 'f1: {:.2f} (train),\t{:.2f} (test)'.format(
            f1_train, f1_test
        )
        print(s, end='', flush=True)

        f1_train_list.append(f1_train * 100.)
        f1_test_list.append(f1_test * 100.)
        ax[1].plot(epochs, f1_train_list, label='f1-score: train', color=colors[0])
        ax[1].plot(epochs, f1_test_list, label='f1-score: test', color=colors[1])
        if not epoch:
            ax[1].legend(loc='best', frameon=False)

        plt.savefig('evolution.png', dpi=300)
        plt.show(block=False)
        fig.canvas.flush_events()

        np.random.shuffle(best_individuals)
        np.random.shuffle(other_individuals)
        # Новая кровь
        for i in range(len(other_individuals) - n_new, len(other_individuals)):
            other_individuals[i] = MLP(input_layer_size=4, output_layer_size=3, hidden_layer_sizes=None)
        # Скрещивание
        print('{} | crossover'.format(s), end='', flush=True)
        for individual in other_individuals[-n_new:]:
            best_parent = np.random.choice(best_individuals)
            individual.crossover(best_parent, percentage=np.random.rand())
            # individual.crossover(best_parent, percentage=0.5)
        # Мутация
        print('{} | mutation'.format(s), end='', flush=True)
        for individual in other_individuals[-n_new:]:
            individual.mutation(n=n_genes_mutable)
        print(s, end='', flush=True)
        new_population = best_individuals.tolist() + other_individuals.tolist()
        np.random.shuffle(new_population)
        population = new_population

    plt.show()
    plt.ioff()
    plt.close()
