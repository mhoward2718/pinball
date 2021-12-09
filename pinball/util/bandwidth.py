"""Methods for computing bandwitdh.

    * Bofinger, E. (1975). Estimation of a density function using order statistics. Australian Journal of Statistics 17: 1-17.
    * Chamberlain, G. (1994). Quantile regression, censoring, and the structure of wages. In Advances in Econometrics, Vol. 1: Sixth World Congress, ed. C. A. Sims, 171-209. Cambridge: Cambridge University Press.
    * Hall, P., and S. Sheather. (1988). On the distribution of the Studentized quantile. Journal of the Royal Statistical Society, Series B 50: 381-391.
"""
from scipy.stats import norm

def hall_sheather(n: float, q: float, alpha:float = .05) -> float:
    """Hall-Sheather bandwidth.
    
    Copy of implementation in StatsModels.
    TODO: Provide link to file for reference implementation
    
    Args:
    
    Returns:
    
    """
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z)**2.
    den = 2. * z**2. + 1.
    h = n**(-1. / 3) * norm.ppf(1. - alpha / 2.)**(2./3) * (num / den)**(1./3)
    return h


def bofinger(n: float, q: float) -> float:
    """bofinger bandwidth.
    
    Copy of implementation in StatsModels.
    TODO: Provide link to file for reference implementation
    
    Args:
    
    Returns:
    
    """
    # Possibly different from quantreg implementation
    num = 9. / 2 * norm.pdf(2 * norm.ppf(q))**4
    den = (2 * norm.ppf(q)**2 + 1)**2
    h = n**(-1. / 5) * (num / den)**(1. / 5)
    return h


def chamberlain(n: float, q: float, alpha: float=.05) -> float:
    """chamberlain bandwidth.
    
    Copy of implementation in StatsModels.
    TODO: Provide link to file for reference implementation
    
    Args:
    
    Returns:
    
    """
    return norm.ppf(1 - alpha / 2) * np.sqrt(q*(1 - q) / n)