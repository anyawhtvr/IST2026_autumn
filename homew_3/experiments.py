"""
Код экспериментов для отчёта (задания 3, 4, 7, 8).
Отчёт homework-practice-03-Optimization_in_ML.ipynb вызывает функции отсюда,
чтобы в самом отчёте было мало кода. Все эксперименты воспроизводимы
(фиксированный seed).
"""
import time

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

import oracles
import optimization
from plot_trajectory_2d import plot_levels, plot_trajectory


# ---------------------------------------------------------------------------
# Задание 3: траектории градиентного спуска на двумерных квадратичных функциях
# ---------------------------------------------------------------------------

def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


# Три квадратичные функции f(x) = 1/2 x^T A x (b = 0, минимум в нуле)
TASK3_FUNCTIONS = {
    r'$\kappa = 2$ (хорошо обусловленная)': np.diag([1.0, 2.0]),
    r'$\kappa = 20$ (плохо обусловленная)': np.diag([1.0, 20.0]),
    r'$\kappa = 9$, оси повёрнуты на 45°':
        _rotation(np.pi / 4) @ np.diag([1.0, 9.0]) @ _rotation(np.pi / 4).T,
}

TASK3_STARTS = [np.array([-5.0, 4.0]), np.array([5.0, 1.0]),
                np.array([5.0, 0.0])]
TASK3_COLORS = ['tab:red', 'tab:blue', 'tab:green']
TASK3_LEVELS = [0.25, 1, 4, 9, 16, 25, 49, 81, 144]


def task3_strategies(A):
    """Стратегии выбора шага; константный шаг 1/L, L = lambda_max(A)."""
    L = np.linalg.eigvalsh(A).max()
    return {
        'Constant, c = 1/L': {'method': 'Constant', 'c': 1.0 / L},
        'Constant, c = 1.9/L': {'method': 'Constant', 'c': 1.9 / L},
        'Armijo': {'method': 'Armijo'},
        'Wolfe': {'method': 'Wolfe'},
    }


def run_task3(tolerance=1e-8, max_iter=10000):
    """Рисует сетку графиков (функции x стратегии) и возвращает таблицу
    числа итераций: rows[(function, strategy)] = [iters for each start]."""
    n_rows = len(TASK3_FUNCTIONS)
    n_cols = len(task3_strategies(np.eye(2)))
    fig = plt.figure(figsize=(5.0 * n_cols, 4.6 * n_rows))
    table = {}
    for i, (fname, A) in enumerate(TASK3_FUNCTIONS.items()):
        oracle = oracles.QuadraticOracle(A, np.zeros(2))
        for j, (sname, ls) in enumerate(task3_strategies(A).items()):
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            plot_levels(oracle.func, xrange=[-6, 6], yrange=[-5, 5],
                        levels=TASK3_LEVELS)
            iters = []
            for x0, color in zip(TASK3_STARTS, TASK3_COLORS):
                _, msg, hist = optimization.gradient_descent(
                    oracle, x0, tolerance=tolerance, max_iter=max_iter,
                    line_search_options=ls, trace=True)
                n_it = len(hist['x']) - 1
                iters.append(n_it if msg == 'success' else msg)
                plot_trajectory(oracle.func, hist['x'],
                                label='$x_0$ = ({:g}, {:g}): {} итер.'.format(
                                    x0[0], x0[1], n_it))
                # plot_trajectory рисует всё красным -- перекрашиваем линию
                line = plt.gca().lines[-1]
                line.set_color(color)
                line.set_linewidth(2.5)
                line.set_markersize(6)
            table[(fname, sname)] = iters
            plt.xlim(-6, 6)
            plt.ylim(-5, 5)
            plt.gca().set_aspect('equal')
            plt.title('{}\n{}'.format(fname, sname), fontsize=11)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.show()
    return table


# ---------------------------------------------------------------------------
# Задание 4: число итераций T(n, kappa)
# ---------------------------------------------------------------------------

def random_quadratic(n, kappa, rng):
    """Случайная квадратичная задача: A = Diag(a), a_i ~ U[1, kappa],
    min(a) = 1, max(a) = kappa; b ~ N(0, I)."""
    a = rng.uniform(1.0, kappa, size=n)
    a[0] = 1.0
    a[-1] = kappa
    b = rng.standard_normal(n)
    return oracles.QuadraticOracle(scipy.sparse.diags(a), b)


def run_task4(ns, kappas, n_seeds, method, tolerance=1e-6, max_iter=100000,
              seed=0):
    """Возвращает массив T[n_idx, seed, kappa_idx] числа итераций GD."""
    rng = np.random.default_rng(seed)
    T = np.zeros((len(ns), n_seeds, len(kappas)))
    for i, n in enumerate(ns):
        for s in range(n_seeds):
            for k, kappa in enumerate(kappas):
                oracle = random_quadratic(n, kappa, rng)
                _, msg, hist = optimization.gradient_descent(
                    oracle, np.zeros(n), tolerance=tolerance,
                    max_iter=max_iter, trace=True,
                    line_search_options={'method': method})
                T[i, s, k] = len(hist['func']) - 1 if msg == 'success' \
                    else np.nan
    return T


def plot_task4(ns, kappas, results, colors=None):
    """results: dict method -> T из run_task4. Один подграфик на метод."""
    if colors is None:
        colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple',
                  'tab:orange']
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (method, T) in zip(axes, results.items()):
        for i, n in enumerate(ns):
            for s in range(T.shape[1]):
                ax.plot(kappas, T[i, s], color=colors[i], linewidth=2,
                        alpha=0.55, label='n = {}'.format(n) if s == 0
                        else None)
        ax.set_xlabel(r'число обусловленности $\kappa$')
        ax.set_ylabel(r'число итераций $T(n, \kappa)$')
        ax.set_title('Градиентный спуск, линейный поиск: {}'.format(method))
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Задание 7: проверка оракула разностными производными
# ---------------------------------------------------------------------------

def check_oracle_finite_diff(m=60, n=5, regcoef=0.1, n_points=3, seed=0):
    """Сравнивает grad/hess оракула с разностными аппроксимациями."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    b = rng.choice([-1.0, 1.0], size=m)
    rows = []
    for kind, matrix in [('dense', A),
                         ('sparse', scipy.sparse.csr_matrix(A))]:
        oracle = oracles.LogRegL2Oracle(matrix, b, regcoef)
        for p in range(n_points):
            x = rng.standard_normal(n) * (p + 1)
            g, g_fd = oracle.grad(x), oracles.grad_finite_diff(oracle.func, x)
            H, H_fd = oracle.hess(x), oracles.hess_finite_diff(oracle.func, x)
            rows.append((kind, p, np.linalg.norm(x),
                         np.max(np.abs(g - g_fd)),
                         np.max(np.abs(H - H_fd)),
                         np.max(np.abs(g)), np.max(np.abs(H))))
    print('{:>7} {:>6} {:>8} {:>14} {:>14} {:>10} {:>10}'.format(
        'A', 'точка', '||x||', 'max|g-g_fd|', 'max|H-H_fd|', 'max|g|',
        'max|H|'))
    for r in rows:
        print('{:>7} {:>6} {:>8.2f} {:>14.2e} {:>14.2e} {:>10.3f} {:>10.3f}'
              .format(*r))


# ---------------------------------------------------------------------------
# Задание 8: GD vs Newton на реальных данных
# ---------------------------------------------------------------------------

def load_dataset(path):
    A, b = load_svmlight_file(path)
    return A.tocsr(), b


def run_task8(A, b, methods=('gd', 'newton')):
    """Запускает методы с параметрами по умолчанию, x_0 = 0, lambda = 1/m."""
    m, n = A.shape
    oracle = oracles.LogRegL2Oracle(A, b, 1.0 / m)
    x0 = np.zeros(n)
    runners = {'gd': ('Градиентный спуск', optimization.gradient_descent),
               'newton': ('Метод Ньютона', optimization.newton)}
    results = {}
    for key in methods:
        name, method = runners[key]
        t = time.time()
        _, msg, hist = method(oracle, x0, trace=True)
        results[name] = hist
        print('{:>18}: {:>20}, итераций: {:>5}, время: {:7.2f} с, '
              'f* = {:.6f}'.format(name, msg, len(hist['func']) - 1,
                                   time.time() - t, float(hist['func'][-1])))
    return results


def plot_task8(results, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    styles = {'Градиентный спуск': 'tab:blue', 'Метод Ньютона': 'tab:red'}
    for name, hist in results.items():
        t = np.array(hist['time'])
        f = np.array(hist['func'], dtype=float).ravel()
        g = np.array(hist['grad_norm']) ** 2
        axes[0].plot(t, f, '-', color=styles[name], linewidth=2.5,
                     label=name, marker='o' if len(t) < 40 else None,
                     markersize=4)
        axes[1].semilogy(t, g / g[0], '-', color=styles[name], linewidth=2.5,
                         label=name, marker='o' if len(t) < 40 else None,
                         markersize=4)
    axes[0].set_ylabel('$f(x_k)$')
    axes[0].set_title('{}: значение функции'.format(title))
    axes[1].set_ylabel(r'$\|\nabla f(x_k)\|_2^2 \,/\, \|\nabla f(x_0)\|_2^2$')
    axes[1].set_title('{}: относительный квадрат нормы градиента'
                      .format(title))
    for ax in axes:
        ax.set_xlabel('время работы, с')
        ax.grid(True, which='both', alpha=0.4)
        ax.legend()
    plt.tight_layout()
    plt.show()
