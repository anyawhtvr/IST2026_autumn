import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Parameters
    ----------
        A : np.array or scipy.sparse.csr_matrix of shape (m, n)
            Feature matrix (training sample).
        b : np.array of shape (m,)
            Labels vector with values in {-1, 1}.
        regcoef : float
            Regularization coefficient.
    """
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self._m = A.shape[0]

    # Sparse matrices denser than this are multiplied as dense ones when
    # computing the Hessian (BLAS is much faster than sparse x sparse there).
    _DENSE_HESS_THRESHOLD = 0.1

    def _margins(self, x):
        # z = b * (A x), elementwise. The last computed value is cached, so
        # consecutive func(x) and grad(x) at the same point cost one A.dot(x).
        cache = getattr(self, '_cache', None)
        if (cache is not None and cache[0].shape == np.shape(x)
                and np.array_equal(cache[0], x)):
            return cache[1]
        z = self.b * self.A.dot(x)
        self._cache = (np.array(x, copy=True), z)
        return z

    def _hess_matrix(self):
        """Matrix A used for the Hessian product (dense for dense-ish A)."""
        if not scipy.sparse.issparse(self.A):
            return self.A
        density = self.A.nnz / float(np.prod(self.A.shape))
        if density < self._DENSE_HESS_THRESHOLD:
            return self.A
        if getattr(self, '_A_dense', None) is None:
            self._A_dense = self.A.toarray()
        return self._A_dense

    def func(self, x):
        # f(x) = 1/m * 1^T ln(1 + exp(-z)) + regcoef/2 * ||x||^2
        z = self._margins(x)
        return (np.mean(np.logaddexp(0, -z))
                + 0.5 * self.regcoef * np.dot(x, x))

    def grad(self, x):
        # grad f(x) = -1/m * A^T (b * sigma(-z)) + regcoef * x
        z = self._margins(x)
        return (-self.A.T.dot(self.b * expit(-z)) / self._m
                + self.regcoef * x)

    def hess(self, x):
        # hess f(x) = 1/m * A^T Diag(s) A + regcoef * I,
        # s = sigma(-z) * (1 - sigma(-z)) = sigma(z) * sigma(-z)
        z = self._margins(x)
        s = expit(z) * expit(-z)
        A = self._hess_matrix()
        if scipy.sparse.issparse(A):
            SA = scipy.sparse.csr_matrix(A.multiply(s[:, None]))
            H = A.T.dot(SA)
            H = H.toarray() if scipy.sparse.issparse(H) else np.asarray(H)
        else:
            H = A.T.dot(A * s[:, None])
        H = H / self._m
        H[np.diag_indices_from(H)] += self.regcoef
        return H


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f_x = func(x)
    result = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        result[i] = (func(x + eps * e_i) - f_x) / eps
    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f_x = func(x)
    E = np.eye(n) * eps
    # f(x + eps * e_i) for every i
    f_shift = np.array([func(x + E[i]) for i in range(n)])
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            value = (func(x + E[i] + E[j]) - f_shift[i] - f_shift[j]
                     + f_x) / eps ** 2
            result[i, j] = value
            result[j, i] = value
    return result
