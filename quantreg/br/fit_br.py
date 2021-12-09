"""Regression quantiles using original simplex approach
 of Barrodale-Roberts/Koenker-d'Orey. 
 
 BR is suitable generally for problems of 10,000 observations or fewer.
 
 The BR Fortran routine can be used to fit a single quantile, or for all rq solutions.
 
 # I hate multiple modes. Begs to be separate classes or mixin
 # TODO: Add references two rank inversion confidence intervals as part of method docstrings
 # This large comment should be part of the docstring of fit_br
 Basically there are two modes of use:
 1.  For Single Quantiles:

       if tau is between 0 and 1 then only one quantile solution is computed.

       if ci = FALSE  then just the point estimate and residuals are returned
		If the column dimension of x is 1 then ci is set to FALSE since
		since the rank inversion method has no proper null model.
       if ci = TRUE  then there are two options for confidence intervals:

               1.  if iid = TRUE we get the original version of the rank
                       inversion intervals as in Koenker (1994)
               2.  if iid = FALSE we get the new version of the rank inversion
                       intervals which accounts for heterogeneity across
                       observations in the conditional density of the response.
                       The theory of this is described in Koenker-Machado(1999)
               Both approaches involve solving a parametric linear programming
               problem, the difference is only in the factor qn which
               determines how far the PP goes. In either case one can
               specify two other options:
                       1. interp = FALSE 
                               returns two intervals an upper and a
                               lower corresponding to a level slightly
                               above and slightly below the one specified
                               by the parameter alpha and dictated by the
                               essential discreteness in the test statistic.
                          interp = TRUE
                               returns a single interval based on
                               linear interpolation of the two intervals
                               returned:  c.values and p.values which give
                               the critical values and p.values of the
                               upper and lower intervals. 
                               Default: interp = TRUE.
                       2.  tcrit = TRUE uses Student t critical values while
                               tcrit = FALSE uses normal theory critical values.
 2. For Multiple Quantiles:
    When tau is None, the the program computes all quantile regression 
    solutions as a process in tau, the resulting arrays
    containing the primal and dual solutions, sol and dsol. 
    These arrays aren't printed by the default
    print function but they are available as attributes.
    This solution is memory and cpu intensive.  On typical machines it is
    not recommended for problems with n > 10,000.
    In large problems a grid of solutions is probably sufficient.

"""

from typing import Callable
import numpy as np
from scipy.stats import t, norm
from statsmodels.regression.linear_model import WLS
from collections import namedtuple
from quantreg.util.bandwidth import hall_sheather
from quantreg_native import rqbr

BRParams = namedtuple("BRParams",
                        ['m',
                        'nn',
                        'm5',
                        'n3',
                        'n4',
                        'a',
                        'b',
                        't',
                        'toler',
                        'ift',
                        'x',
                        'e',
                        's',
                        'wa',
                        'wb',
                        'nsol',
                        'ndsol',
                        'sol',
                        'dsol',
                        'lsol',
                        'h',
                        'qn',
                        'cutoff',
                        'ci',
                        'tnmat',
                        'big',
                        'lci1'])

BRSolution = namedtuple("BRSolution",
                         ['flag',
                          'coef',
                          'resid',
                          'sol',
                          'dsol',
                          'lsol',
                          'h',
                          'qn',
                          'cutoff',
                          'ci',
                          'tnmat'])
    
def derive_br_params(X: np.array, y: np.array, tau: float) -> BRParams:
    """Get parameter tuple for BR solver.
    """
    (n, p) = X.shape
    nsol = 2
    ndsol = 2
    t = tau
    
    if not tau: # This is the multi-quantile case
        nsol = 3 * n
        ndsol = 3 * n
        t = -1    
    
    params = BRParams(m=n,
                    nn=np.int32(p),
                    m5=np.int32(n + 5), 
                    n3=np.int32(p + 3),
                    n4=np.int32(p + 4),
                    a=X,
                    b=y,
                    t=t,
                    toler=np.finfo(np.float64).eps ** (2/3),
                    ift=np.int32(1), 
                    x=np.zeros(p, np.float64),
                    e=np.zeros(n, np.float64),
                    s=np.zeros(n,dtype=np.int32),
                    wa=np.zeros(((n + 5),(p + 4)), dtype=np.float64),
                    wb=np.zeros(n, dtype=np.float32),
                    nsol=np.int32(nsol),
                    ndsol=np.int32(ndsol),
                    sol=np.zeros(((p + 3), nsol), dtype=np.float64),
                    dsol=np.zeros((n, ndsol), dtype=np.float64),
                    lsol=np.int32(0),
                    h=np.zeros((p,nsol), dtype=np.int32),
                    qn=np.zeros(p, dtype=np.float64), # In cases where were generate CI, this value is updated later
                    cutoff=np.float64(0), # In cases where were generate CI, this value is updated later
                    ci=np.zeros((4,p), dtype=np.float64),
                    tnmat=np.zeros((4,p), dtype=np.float64),
                    big=np.finfo(np.float64).max,
                    lci1=np.bool_(False) # In cases where we generate CI, this value is updated later
                    )
    return params
       
def get_wls_weights(X: np.array,
                    y: np.array,
                    tau: float,
                    eps: float = np.finfo(np.float64).eps ** (2/3),
                    bandwidth: Callable=hall_sheather) -> np.array:
    """Get weights for weighted least squares regression step in QN estimation.
    
    TODO: Fill more details about what this is and why we do it
    
    """
    (m, _) = X.shape
    h = bandwidth(tau, m)
    bhi = fit_br(X, y, tau=tau + h, ci=False).coef
    blo = fit_br(X, y, tau=tau - h, ci=False).coef
    dyhat = np.matmul(X,bhi-blo)
    if np.any(dyhat <= 0): 
        pfis = (100 * np.sum(dyhat <= 0))/m
        print("Percent fis <= 0: " + str(pfis))
    return np.maximum(eps, (2 * h)/(dyhat - eps))

def get_qn(X: np.array, 
           y: np.array,
           tau: float,
           iid: bool,
           bandwidth: Callable) -> np.array:
    (m, nn) = X.shape
    qn = np.zeros(nn)
    if not iid:
        weights = get_wls_weights(X, y, tau, m, bandwidth)
        for j in range(0, nn):
            x1 = np.delete(X,j,axis=1)
            qnj = WLS(X[:, j], x1, weights = weights).fit().resid
            qn[j] = np.sum(qnj * qnj)
    else: # Get QN for IID case
        X_cross = np.matmul(X.T, X)
        qn = 1/np.diag(np.linalg.inv(X_cross))
        
    return qn

def fit_br(X: np.array,
           y: np.array, 
           tau:float = 0.5,
           alpha:float = 0.1,
           ci:bool = False,
           iid:bool = True,
           interp:bool = True,
           tcrit:bool = True,
           bandwidth: Callable = hall_sheather) -> BRSolution:
    """Fit quantile regression using Koener-d'Orey adaptation of Barrodale-Roberts solver.
    """
    
    if np.linalg.cond(X) >= 1/(np.finfo(X.dtype).eps):
        raise ValueError('Singular design matrix') 
        
    br_params = derive_br_params(X, y, tau)

    if tau: # Single quantile case 
        if br_params.nn == 1: # Always generate confidence intervals for single variable case
            ci = True
        if ci: # if tau and (ci or (br_params.nn == 1))
            if tcrit:
                cutoff = t.ppf(1 - alpha/2, br_params.m - br_params.nn)
            else:
                cutoff = norm.ppf(1 - alpha/2)
            qn = get_qn(X, y, tau, cutoff, iid, bandwidth)
            br_params = br_params._replace(lci1=True, qn=qn, cutoff=cutoff)
    
    solution = BRSolution(*rqbr(**br_params._asdict()))
    
    if solution.flag:
        print("Solution may be nonunique. Premature end - possible conditioning problem in x")

    return solution