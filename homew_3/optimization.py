import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
import warnings
from datetime import datetime
from collections import defaultdict


def _is_bad(value):
    """True if value is None or contains inf/nan (or cannot be checked)."""
    if value is None:
        return True
    try:
        return not np.all(np.isfinite(value))
    except TypeError:
        return True


def _record(history, start_time, func_k, grad_norm_k, x_k):
    """Appends the current state of the method into history (if tracing)."""
    if history is None:
        return
    history['time'].append((datetime.now() - start_time).total_seconds())
    history['func'].append(np.copy(func_k))
    history['grad_norm'].append(grad_norm_k)
    if x_k.size <= 2:
        history['x'].append(np.copy(x_k))


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        if self._method == 'Wolfe':
            alpha = None
            try:
                with warnings.catch_warnings():
                    # scipy warns when the line search does not converge;
                    # we handle this case ourselves by falling back to Armijo.
                    warnings.simplefilter('ignore')
                    result = scipy.optimize.line_search(
                        oracle.func, oracle.grad, x_k, d_k,
                        c1=self.c1, c2=self.c2)
                alpha = getattr(result, 'alpha', None)
                if alpha is None:
                    alpha = result[0]
            except (ValueError, FloatingPointError, OverflowError,
                    ZeroDivisionError):
                alpha = None
            if alpha is not None:
                return alpha
            # Fallback: backtracking procedure for the Armijo condition.

        return self._armijo(oracle, x_k, d_k, previous_alpha)

    def _armijo(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Backtracking procedure (Algorithm 2): halves alpha until the Armijo
        condition phi(alpha) <= phi(0) + c1 * alpha * phi'(0) holds.
        """
        alpha = previous_alpha if previous_alpha is not None else self.alpha_0
        phi_0 = oracle.func_directional(x_k, d_k, 0.0)
        derphi_0 = oracle.grad_directional(x_k, d_k, 0.0)
        # Guard against an infinite loop on degenerate inputs.
        while (oracle.func_directional(x_k, d_k, alpha)
               > phi_0 + self.c1 * alpha * derphi_0):
            alpha /= 2.0
            if alpha < 1e-300:
                break
        return alpha


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict or None
        Dictionary with line search options, e.g. {'method': 'Armijo', 'c1': 1e-4}.
        See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    grad_norm_0_sq = None
    alpha_k = None

    for k in range(max_iter + 1):
        # (Oracle call)
        with np.errstate(all='ignore'):
            func_k = oracle.func(x_k)
            grad_k = oracle.grad(x_k)
        if _is_bad(x_k) or _is_bad(func_k) or _is_bad(grad_k):
            if display:
                print('Iteration {}: computational error'.format(k))
            return x_k, 'computational_error', history

        grad_norm_k_sq = np.dot(grad_k, grad_k)
        grad_norm_k = np.sqrt(grad_norm_k_sq)
        if grad_norm_0_sq is None:
            grad_norm_0_sq = grad_norm_k_sq
        _record(history, start_time, func_k, grad_norm_k, x_k)
        if display:
            print('Iteration {}: f = {}, ||grad|| = {:.3e}'.format(
                k, np.squeeze(func_k), grad_norm_k))

        # (Stopping criterion (1.1))
        if grad_norm_k_sq <= tolerance * grad_norm_0_sq:
            return x_k, 'success', history
        if k == max_iter:
            break

        # (Direction and line search)
        d_k = -grad_k
        previous_alpha = 2.0 * alpha_k if alpha_k is not None else None
        with np.errstate(all='ignore'):
            alpha_k = line_search_tool.line_search(
                oracle, x_k, d_k, previous_alpha=previous_alpha)
        if _is_bad(alpha_k):
            if display:
                print('Iteration {}: line search failed'.format(k))
            return x_k, 'computational_error', history

        # (Update)
        x_k = x_k + alpha_k * d_k

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict or None
        Dictionary with line search options, e.g. {'method': 'Armijo', 'c1': 1e-4}.
        See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    grad_norm_0_sq = None

    for k in range(max_iter + 1):
        # (Oracle call)
        with np.errstate(all='ignore'):
            func_k = oracle.func(x_k)
            grad_k = oracle.grad(x_k)
        if _is_bad(x_k) or _is_bad(func_k) or _is_bad(grad_k):
            if display:
                print('Iteration {}: computational error'.format(k))
            return x_k, 'computational_error', history

        grad_norm_k_sq = np.dot(grad_k, grad_k)
        grad_norm_k = np.sqrt(grad_norm_k_sq)
        if grad_norm_0_sq is None:
            grad_norm_0_sq = grad_norm_k_sq
        _record(history, start_time, func_k, grad_norm_k, x_k)
        if display:
            print('Iteration {}: f = {}, ||grad|| = {:.3e}'.format(
                k, np.squeeze(func_k), grad_norm_k))

        # (Stopping criterion (1.1))
        if grad_norm_k_sq <= tolerance * grad_norm_0_sq:
            return x_k, 'success', history
        if k == max_iter:
            break

        # (Newton direction: solve hess * d = -grad via Cholesky)
        with np.errstate(all='ignore'):
            hess_k = oracle.hess(x_k)
        if scipy.sparse.issparse(hess_k):
            hess_k = hess_k.toarray()
        hess_k = np.atleast_2d(hess_k)
        if _is_bad(hess_k):
            if display:
                print('Iteration {}: computational error in Hessian'.format(k))
            return x_k, 'computational_error', history
        try:
            c_and_lower = scipy.linalg.cho_factor(hess_k)
            d_k = scipy.linalg.cho_solve(c_and_lower, -np.atleast_1d(grad_k))
        except (LinAlgError, ValueError):
            if display:
                print('Iteration {}: Hessian is not positive definite'.format(k))
            return x_k, 'newton_direction_error', history
        d_k = d_k.reshape(np.shape(x_k))

        # (Line search: always start from the unit step)
        with np.errstate(all='ignore'):
            alpha_k = line_search_tool.line_search(
                oracle, x_k, d_k, previous_alpha=1.0)
        if _is_bad(alpha_k):
            if display:
                print('Iteration {}: line search failed'.format(k))
            return x_k, 'computational_error', history

        # (Update)
        x_k = x_k + alpha_k * d_k

    return x_k, 'iterations_exceeded', history
