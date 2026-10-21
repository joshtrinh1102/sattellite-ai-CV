"""Cross-component analysis - Component 3 Stage B.

Joins the occupancy time series (Component 1) to the risk-topic time series
(Components 2 and 3) to answer the core research question: of the candidate
factors affecting port traffic, which ones actually do.

WHAT THIS IMPLEMENTS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------------
3_ranking/plan-topics.md ties the method to the number of capture dates, and that
table is binding, not advisory:

    ~10 dates  two-group comparison only
    ~30 dates  2-3 pre-specified hypotheses
    ~60 dates  regularised regression over all 12 topics, ranked

`choose_method` reads the actual row count and refuses to run a method the
sample cannot support, because - in the plan's words - running the 60-date
method on 10 dates "would be fabrication dressed as a result".

Everything here is numpy + scipy. scikit-learn and statsmodels are installed
but their compiled extensions are blocked by this machine's Application
Control policy, so ridge, the elastic net and the bootstrap are implemented
directly. They are short and the formulas are standard.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Regression primitives
# ---------------------------------------------------------------------------

def standardise(X):
    """Centre and scale columns; return (Z, mean, std). Zero-variance columns
    are left at zero rather than dividing by zero - a topic that never appears
    in the window carries no information and should contribute nothing."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    keep = sd > 1e-12
    Z = np.zeros_like(X, dtype="float64")
    Z[:, keep] = (X[:, keep] - mu[keep]) / sd[keep]
    return Z, mu, np.where(keep, sd, 1.0)


def ols_with_ci(X, y, alpha=0.05):
    """Plain least squares with an intercept, returning coefficients and CIs.

    Used for the small pre-specified models, where the point is an effect size
    with an interval around it rather than a ranking.
    """
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = n - A.shape[1]
    if dof <= 0:
        raise ValueError(f"{n} observations cannot fit {A.shape[1]} parameters")
    sigma2 = resid @ resid / dof
    XtX_inv = np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.maximum(0, np.diag(XtX_inv) * sigma2))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), dof))
    return {
        "beta": beta, "se": se, "t": tstat, "p": pval,
        "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se,
        "dof": dof, "r2": 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())),
    }


def elastic_net(Z, y, lam, l1_ratio=0.5, n_iter=2000, tol=1e-7, penalty=None):
    """Coordinate-descent elastic net on standardised Z and centred y.

    Standard soft-thresholding update. Z must be standardised, which makes the
    per-column denominator constant and the penalty comparable across topics -
    without that, a topic measured on a wider scale gets shrunk less and rises
    up the ranking for purely dimensional reasons.
    """
    n, p = Z.shape
    beta = np.zeros(p)
    r = y - Z @ beta
    col_ss = (Z ** 2).sum(axis=0)
    pf = np.ones(p) if penalty is None else np.asarray(penalty, dtype="float64")
    l1, l2 = lam * l1_ratio * pf, lam * (1 - l1_ratio) * pf
    for _ in range(n_iter):
        delta = 0.0
        for j in range(p):
            if col_ss[j] <= 1e-12:
                continue
            rho = Z[:, j] @ (r + Z[:, j] * beta[j]) / n
            new = np.sign(rho) * max(abs(rho) - l1[j], 0.0) / (col_ss[j] / n + l2[j])
            if new != beta[j]:
                r -= Z[:, j] * (new - beta[j])
                delta = max(delta, abs(new - beta[j]))
                beta[j] = new
        if delta < tol:
            break
    return beta


def kfold_indices(n, k=5, seed=0):
    """Plain K-fold. NOT shuffled by time on purpose - see `blocked_folds`."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def blocked_folds(n, k=5):
    """Contiguous time blocks.

    Satellite dates are a time series: neighbouring captures share weather,
    news cycle and season. A shuffled fold puts near-duplicate dates on both
    sides of the split and reports an optimistic error, so the lambda chosen
    would be too small and the ranking too long. Blocks keep the split honest.
    """
    bounds = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(k)]


def cv_elastic_net(Z, y, lambdas=None, l1_ratio=0.5, k=5, penalty=None):
    """Pick lambda by blocked CV; return (best_lambda, curve)."""
    lambdas = np.logspace(-3, 0.5, 30) if lambdas is None else lambdas
    folds = blocked_folds(len(y), k=k)
    curve = []
    for lam in lambdas:
        errs = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(y)), f)
            if len(tr) < 3 or len(f) == 0:
                continue
            ym = y[tr].mean()
            b = elastic_net(Z[tr], y[tr] - ym, lam, l1_ratio, penalty=penalty)
            pred = Z[f] @ b + ym
            errs.append(float(np.mean((y[f] - pred) ** 2)))
        curve.append((float(lam), float(np.mean(errs)) if errs else np.inf))
    best = min(curve, key=lambda c: c[1])[0]
    return best, curve


def bootstrap_coefficients(Z, y, lam, l1_ratio=0.5, n_boot=400, seed=0, penalty=None):
    """Resample rows to get a selection frequency and an interval per feature.

    With tens of observations the point estimate of any single coefficient is
    unstable. What survives resampling - how often a topic is selected at all,
    and whether its sign holds - is the part worth reporting.
    """
    rng = np.random.default_rng(seed)
    n, p = Z.shape
    draws = np.zeros((n_boot, p))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(idx).size < 4:
            continue
        ym = y[idx].mean()
        draws[b] = elastic_net(Z[idx], y[idx] - ym, lam, l1_ratio, penalty=penalty)
    return {
        "mean": draws.mean(axis=0),
        "selection_rate": (np.abs(draws) > 1e-8).mean(axis=0),
        "ci_low": np.percentile(draws, 2.5, axis=0),
        "ci_high": np.percentile(draws, 97.5, axis=0),
        "sign_consistency": np.maximum((draws > 0).mean(axis=0), (draws < 0).mean(axis=0)),
    }


def penalty_from_weights(weights, p):
    """Importance weights -> per-feature penalty factors (glmnet's penalty.factor).

    Weight 2 halves the shrinkage on that feature, 0.5 doubles it. This is the
    only place a weight can act: multiplying a feature by a constant does
    nothing once it is standardised. The CV then picks lambda for the weighted
    penalty, so weights shift the balance between features, not the overall
    amount of regularisation.
    """
    if weights is None:
        return np.ones(p)
    w = np.asarray(weights, dtype="float64")
    if w.shape != (p,) or np.any(w <= 0):
        raise ValueError(f"need {p} positive weights, got {w.tolist()}")
    return 1.0 / w


def benjamini_hochberg(pvals, q=0.05):
    """BH step-up. Returns a boolean 'survives' array and adjusted p-values."""
    p = np.asarray(pvals, dtype="float64")
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out <= q, out


# ---------------------------------------------------------------------------
# The method gate
# ---------------------------------------------------------------------------

METHOD_TABLE = [
    (60, "elastic_net", "Regularised regression over all topics, ranked by effect size."),
    (25, "prespecified", "Test 2-3 factor hypotheses chosen in advance."),
    (0, "two_group", "COVID vs normal comparison only. Nothing about individual factors."),
]


def choose_method(n_dates):
    """Map the observed sample size onto the plan's sample-size table."""
    for threshold, name, description in METHOD_TABLE:
        if n_dates >= threshold:
            return {"n_dates": n_dates, "method": name, "description": description,
                    "threshold": threshold}
    return {"n_dates": n_dates, "method": "two_group", "threshold": 0,
            "description": METHOD_TABLE[-1][2]}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_factors(X, y, feature_names, l1_ratio=0.5, n_boot=400, seed=0, weights=None):
    """Rank candidate factors by their association with ship density.

    Three views, because no single one is trustworthy at this sample size:

      * `spearman` - univariate, robust to the non-normal topic shares, and
        carrying a BH correction across the whole factor set.
      * `enet_beta` - the multivariate coefficient after regularisation, which
        is what separates a factor with an independent contribution from one
        that merely rides along with a correlated neighbour.
      * `selection_rate` - how often the factor survives at all under
        bootstrap resampling. This is the stability column, and it is the one
        to read first: a large coefficient selected 30% of the time is noise.

    Returns rows sorted by selection rate then |coefficient|.
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    Z, _, _ = standardise(X)

    pf = penalty_from_weights(weights, X.shape[1])
    lam, curve = cv_elastic_net(Z, y, l1_ratio=l1_ratio, penalty=pf)
    beta = elastic_net(Z, y - y.mean(), lam, l1_ratio, penalty=pf)
    boot = bootstrap_coefficients(Z, y, lam, l1_ratio, n_boot=n_boot, seed=seed,
                                  penalty=pf)

    rhos, praw = [], []
    for j in range(X.shape[1]):
        if np.std(X[:, j]) < 1e-12:
            rhos.append(0.0); praw.append(1.0); continue
        r, p = stats.spearmanr(X[:, j], y)
        rhos.append(float(r)); praw.append(float(p))
    survives, padj = benjamini_hochberg(praw)

    rows = []
    for j, name in enumerate(feature_names):
        rows.append({
            "factor": name,
            "weight": float(1.0 / pf[j]),
            "enet_beta": float(beta[j]),
            "selection_rate": float(boot["selection_rate"][j]),
            "sign_consistency": float(boot["sign_consistency"][j]),
            "boot_ci_low": float(boot["ci_low"][j]),
            "boot_ci_high": float(boot["ci_high"][j]),
            "spearman_rho": rhos[j],
            "spearman_p": praw[j],
            "spearman_p_adj": float(padj[j]),
            "bh_survives": bool(survives[j]),
        })
    rows.sort(key=lambda r: (-r["selection_rate"], -abs(r["enet_beta"])))
    return {"lambda": lam, "cv_curve": curve, "ranking": rows,
            "n": len(y), "n_features": X.shape[1]}


def _partial_rank_corr(x, y, C):
    """Spearman correlation of x and y after removing controls C from both.

    Ranks x and y, regresses each on the controls (plus intercept), and
    correlates the residuals. Returns (rho, p) with p from the t distribution on
    n - 2 - k degrees of freedom.
    """
    n = len(y)
    A = np.column_stack([np.ones(n), C]) if C.size else np.ones((n, 1))
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return 0.0, 1.0
    r = float(np.clip(np.corrcoef(ex, ey)[0, 1], -0.999999, 0.999999))
    dof = n - 2 - (A.shape[1] - 1)
    if dof <= 0:
        return r, 1.0
    t = r * np.sqrt(dof / (1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), dof))


def partial_spearman_ranking(X, y, feature_names, test_idx, control_idx,
                             n_boot=1000, seed=0):
    """Exploratory ranking for samples too small for the elastic net.

    One partial Spearman correlation per tested feature, controlling for
    `control_idx` (season and trend), with a bootstrap 95% interval and a BH
    correction across the tested features only. Univariate by construction: it
    cannot separate two dimensions that move together, which is exactly what
    the elastic net does once there are enough dates.
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    C = X[:, control_idx] if len(control_idx) else np.empty((len(y), 0))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]

    rows, praw = [], []
    for j in test_idx:
        rho, p = _partial_rank_corr(X[:, j], y, C)
        draws = []
        for idx in boots:
            if np.unique(idx).size < C.shape[1] + 4:
                continue
            draws.append(_partial_rank_corr(X[idx, j], y[idx], C[idx])[0])
        draws = np.asarray(draws) if draws else np.array([rho])
        rows.append({"factor": feature_names[j], "partial_rho": rho,
                     "rho_ci_low": float(np.percentile(draws, 2.5)),
                     "rho_ci_high": float(np.percentile(draws, 97.5)),
                     "spearman_p": p})
        praw.append(p)
    survives, padj = benjamini_hochberg(np.array(praw)) if praw else ([], [])
    for r, s, pa in zip(rows, survives, padj):
        r["spearman_p_adj"] = float(pa)
        r["bh_survives"] = bool(s)
    rows.sort(key=lambda r: -abs(r["partial_rho"]))
    return {"method": "partial_spearman", "ranking": rows, "n": n,
            "n_features": len(rows),
            "controls": [feature_names[j] for j in control_idx]}


def explain_date(X, y, feature_names, target_index, lam=None, l1_ratio=0.5,
                 weights=None):
    """Decompose one date's predicted density into per-factor contributions.

    This answers "which factors drove the ship count on THIS date" as opposed
    to "which factors matter on average". For a linear model the decomposition
    is exact: contribution_j = beta_j * z_j, and the contributions plus the
    intercept reconstruct the prediction.

    Read it as attribution under the fitted model, not as a cause. A factor
    only appears here if it was selected globally, so a date driven by
    something outside the feature set shows up as a large residual - which is
    reported alongside, and is the honest signal that the model is missing it.
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    Z, mu, sd = standardise(X)
    pf = penalty_from_weights(weights, X.shape[1])
    if lam is None:
        lam, _ = cv_elastic_net(Z, y, l1_ratio=l1_ratio, penalty=pf)
    ym = y.mean()
    beta = elastic_net(Z, y - ym, lam, l1_ratio, penalty=pf)

    z = Z[target_index]
    contrib = beta * z
    pred = ym + contrib.sum()
    rows = [{
        "factor": feature_names[j],
        "value": float(X[target_index, j]),
        "z": float(z[j]),
        "beta": float(beta[j]),
        "contribution": float(contrib[j]),
    } for j in range(len(feature_names)) if abs(contrib[j]) > 1e-9]
    rows.sort(key=lambda r: -abs(r["contribution"]))
    return {
        "baseline": float(ym),
        "predicted": float(pred),
        "actual": float(y[target_index]),
        "residual": float(y[target_index] - pred),
        "contributions": rows,
        "lambda": lam,
    }


def two_group_comparison(y_a, y_b, label_a="group A", label_b="group B",
                         n_boot=10000, seed=0):
    """Difference in mean density with a bootstrap CI and Mann-Whitney p.

    The honest method at ~10 dates. Bootstrap rather than a t-interval because
    densities are skewed and the groups are small.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(y_a, dtype="float64")
    b = np.asarray(y_b, dtype="float64")
    diff = a.mean() - b.mean()
    draws = np.array([
        rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_boot)
    ])
    try:
        u_p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        u_p = float("nan")
    return {
        "label_a": label_a, "label_b": label_b,
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff": float(diff),
        "pct_change": float(diff / b.mean() * 100) if b.mean() else float("nan"),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "mannwhitney_p": u_p,
    }
