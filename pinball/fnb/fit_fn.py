"""Quantile regression with Frisch-Newton method.
Looks like a simpler wrapper than BR
"""

from typing import Callable
import numpy as np
from scipy.stats import t, norm
from statsmodels.regression.linear_model import WLS
from collections import namedtuple
from pinball.util.bandwidth import hall_sheather
from pinball_native import rqfnb

FNBParams = namedtuple("FNBParams",
                       ["n","p","a","y","rhs","d","u","beta","eps","wn","wp","nit","info"])

FNBSolution = namedtuple("FNBSolution",   
                     ["a", "c", "rhs", "d", "beta", "eps", "wn", "wp", "nit", "info"])

def fit_fn(X: np.array, y: np.array, tau: float = 0.5, rhs = (1-tau)*X, beta = 0.99995, eps = 1e-06):
    """Quantile regression with Frisch-Newton method
    """
    (n, p) = X.shape
    # TODO: Add shape checks
    # TODO: Add bounds on tau
    d = np.ones(n, dtype=np.float64)
    u = np.ones(n, dtype=np.float64)
    wn = np.zeros(10*n, dtype=np.float64)
    wn[1:n] = (1-tau) # Initial value of dual.
    solution = FNBSolution(*rqfnb(**params._asdict())
    pass
    

"rq.fit.fnb" <-
function (x, y, tau = 0.5, rhs = (1-tau)*apply(x,2,sum), beta = 0.99995, eps = 1e-06)
{
    n <- length(y)
    p <- ncol(x)
    if (n != nrow(x))
        stop("x and y don't match n")
    if (tau < eps || tau > 1 - eps)
        stop("No parametric Frisch-Newton method.  Set tau in (0,1)")
    d   <- rep(1,n)
    u   <- rep(1,n)
    wn <- rep(0,10*n)
    wn[1:n] <- (1-tau) #initial value of dual solution

    z <- .Fortran("rqfnb", as.integer(n), as.integer(p), a = as.double(t(as.matrix(x))),
        c = as.double(-y), rhs = as.double(rhs), d = as.double(d),as.double(u),
        beta = as.double(beta), eps = as.double(eps),
        wn = as.double(wn), wp = double((p + 3) * p),
        nit = integer(3), info = integer(1))
    if (z$info != 0)
        warning(paste("Error info = ", z$info, "in stepy: possibly singular design"))
    coefficients <- -z$wp[1:p]
    names(coefficients) <- dimnames(x)[[2]]
    residuals <- y - x %*% coefficients
    list(coefficients=coefficients, tau=tau, residuals=residuals, nit = z$nit)
}
