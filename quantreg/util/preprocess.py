"""Preprocess is actually not a good name for what this does
"""
import numpy as np

def fit_preprocess(X: np.array,
            y: np.array,
            tau: float = 0.5,
            mm_factor: float = 0.8,
            max_fixups: int = 3,
            eps: float = 1e-6,
            fit_method=fit_fbn):
    (n, p) = X.shape
    # TODO: Add checks on singularity and shape. Probably in a separate method upstream from this one
    m = np.rint(np.sqrt(p) * n ** (2/3)) # Initial sample size
    ifix = 0
    ibad = 0
    not_optimal = True
    # Or do optimal = False, while not optimal
    while not_optimal:
        ibad = ibad + 1
        
        # Sample m rows from X
        if m < n:
            sample_idx = np.random.choice(X.shape[0], m, replace=False)
        else:
            solution = method(X, y, tau) # Why no call to eps? Bug?
            # May want to print some type of warning
            break
        XX = X[sample_idx, :]
        yy = y[sample_idx, :]
        solution = fit_method(XX, yy, tau, eps=eps)
        # TODO: Double check below
        # Noticed that result is lower triangular while R is upper triangular.
        # Does this matter? Probably depends how result is used
        XX_inv = np.linalg.inv(np.cholesky(np.matmul(XX.T, XX)))
        # This results in a scalar? What is meaning of ^2?
        # Or does it square everything in the matrix?
        # To check triangularity question above, test if X %*% XX_INV is the same
        # or if band is the same
        # What is this bandwidth? Can it use the other bandwidths already in package?
        band = np.sqrt(np.matmul(np.matmul(X, XX_inv) ** 2), np.ones(p, dtype=np.float64))
        # Signed errors on whole data using coefficients fit by sample
        r = y - np.matmul(X, solution.coef)
        M = mm_factor * m
        # May need to use max, argmax, maximum
        lo_q = max(1/n, tau - M/(2 * n))
        hi_q = min(tau + M/(2 * n), (n - 1)/n)
        # Quantiled of errors scaled by bandwidth/eps
        kappa = np.quantile(r/np.maximum(eps, band), [lo_q, hi_q])
        s_l = r < band * kappa[0] # Should work as long as indexing is right
        s_u = r > band * kappa[1]
        
        bad_fixups = 0
        while not_optimal and (bad_fixups < max_bad_fixups):
            ifix = ifix + 1
            # Get rows above and below bands
            xx = X[~s_l & ~s_u]
            yy = y[~s_l & ~s_u]
            if np.any(s_l):
                low_glob_x = np.sum(X[s_l], axis=0) # Columnwise sum of X in s_u
                low_glob_y = np.sum(y[s_l]) # Sum of y in s_u
                xx = np.vstack(xx, low_glob_x)
                yy = np.vstack(yy, low_glob_y)
            if np.any(s_u):
                high_glob_x = np.sum(X[s_u], axis=0) # Columnwise sum of X in s_u
                high_glob_y = np.sum(y[s_u]) # Sum of y in s_u
                xx = np.vstack(xx, high_glob_x)
                yy = np.vstack(yy, high_glob_y)
            solution = fit_method(xx, yy, tau, eps)
            # Predict on whole data
            r = y - np.matmul(X, solution.coef)
            
            su_bad = (r < 0) & su
            sl_bad = (r > 0) & sl
            
            if np.any(su_bad) or np.any(sl_bad):
                if np.sum(sl_bad | su_bad) > 0.1 * M:
                    print("too many fixups. Doubling m")
                    m = 2*m
                    break
                s_u = s_u & su_bad
                s_l = s_l & sl_bad
            else:
                not_optimal = False

    return solution