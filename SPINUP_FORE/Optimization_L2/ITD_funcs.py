import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def extended_state(a, v, h_bnds):
    """
    Compute the extended state vector
        x = (a0, x1^l, x1^u, x2^l, x2^u, ..., xI^l, xI^u)
    from category areas a, category volumes v, and thickness
    category boundaries h_bnds.
    """
    I = len(a)
    if len(v) != I:
        raise ValueError("a and v must have the same length")
    if len(h_bnds) != I + 1:
        raise ValueError("h_bnds must have I+1 boundaries for I categories")

    a0 = 1.0 - np.sum(a)
    H_l = h_bnds[:-1]
    H_u = h_bnds[1:]

    x_l = np.zeros(I)
    x_u = np.zeros(I)

    nonzero = a > 0
    x_u[nonzero] = (v[nonzero] - a[nonzero] * H_l[nonzero]) / (H_u[nonzero] - H_l[nonzero])
    x_l[nonzero] = a[nonzero] - x_u[nonzero]

    if np.any(x_u[nonzero] < -1e-10) or np.any(x_l[nonzero] < -1e-10):
        raise ValueError("Negative x_l or x_u: category mean thickness "
                          "falls outside its [H_l, H_u] bounds")

    x = np.zeros(1 + 2 * I)
    x[0] = a0
    x[1::2] = x_l
    x[2::2] = x_u
    return x

def invert_extended_state(x, h_bnds):
    """
    Given the extended state and an array for the bounds compute the areas and volumes
    
    """
    d = len(x)             # dimension of extended state
    cats = int((d-1)/2)    # number of ice categories

    # Check that h_bnds is an array with the correct dimension
    if len(h_bnds) != cats + 1:
        raise ValueError("h_bnds must have cats+1 boundaries")
    
    H_l = h_bnds[:-1]  # lower bounds array
    H_u = h_bnds[1:]   # upper bounds array
    
    a = np.zeros(cats) # Initialize area array
    v = np.zeros(cats) # Initialize volume array

    # Invert from extended state to a and v
    for i in range(cats):
        a[i] = x[2*i+1] + x[2*(i+1)]
        v[i] = H_l[i]*x[2*i+1] + H_u[i]*x[2*(i+1)]
    return a, v


def fit_piecewise_linear_pdf(a, v, h_bnds):
    """Reconstruct a piecewise-linear pdf p(h) = m_i*h + b_i on each category."""
    I = len(a)
    H_l = h_bnds[:-1]
    H_u = h_bnds[1:]

    m = np.zeros(I)
    b = np.zeros(I)

    for i in range(I):
        if a[i] <= 0:
            continue
        Hl, Hu = H_l[i], H_u[i]
        A = np.array([
            [(Hu**2 - Hl**2) / 2.0,      Hu - Hl],
            [(Hu**3 - Hl**3) / 3.0, (Hu**2 - Hl**2) / 2.0]
        ])
        rhs = np.array([a[i], v[i]])
        mi, bi = np.linalg.solve(A, rhs)
        m[i], b[i] = mi, bi

    return m, b


def plot_pdf(m, b, h_bnds, a0=None, n=200, ax=None):
    """Plot the piecewise-linear pdf reconstructed by fit_piecewise_linear_pdf."""
    H_l = h_bnds[:-1]
    H_u = h_bnds[1:]
    I = len(m)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for i in range(I):
        h = np.linspace(H_l[i], H_u[i], n)
        p = m[i] * h + b[i]
        ax.plot(h, p, color="C0", lw=1.8)
        if np.any(p < 0):
            ax.fill_between(h, p, 0, where=(p < 0), color="red", alpha=0.3,
                             label="_nolegend_")
            ax.plot(h, p, color="red", lw=1.8)
        ax.axvline(H_u[i], color="gray", linestyle="--", linewidth=0.7)

    ax.axvline(H_l[0], color="gray", linestyle="--", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ice thickness $h$ (m)")
    ax.set_ylabel("$p(h)$")
    ax.set_title("Piecewise-linear ITD density reconstruction")

    if a0 is not None:
        ax.annotate(f"$a_0$ = {a0:.3f}\n(open water, mass at $h=0$)",
                     xy=(0, 0), xytext=(0.03, 0.85),
                     textcoords="axes fraction", fontsize=9,
                     bbox=dict(boxstyle="round", fc="white", ec="gray"))
    return ax


def plot_extended_state(x, h_bnds, ax=None, eps=None):
    """Plot the extended state vector as a discrete distribution over h."""
    x = np.asarray(x, dtype=float)
    h_bnds = np.asarray(h_bnds, dtype=float)

    I = (len(x) - 1) // 2
    if len(x) != 1 + 2 * I:
        raise ValueError("x must have length 1 + 2*I")
    if len(h_bnds) != I + 1:
        raise ValueError("h_bnds must have I+1 boundaries for I categories")

    a0 = x[0]
    x_l = x[1::2]
    x_u = x[2::2]

    if eps is None:
        span = h_bnds[-1] - h_bnds[0]
        eps = 0.01 * span

    locs = [0.0]
    masses = [a0]
    labels = ["a0"]

    for i in range(I):
        Hl, Hu = h_bnds[i], h_bnds[i + 1]
        locs.append(Hl + eps)
        masses.append(x_l[i])
        labels.append(f"x{i+1}^l")
        if i < I - 1:
            locs.append(Hu - eps)
        else:
            locs.append(Hu)
        masses.append(x_u[i])
        labels.append(f"x{i+1}^u")

    locs = np.array(locs)
    masses = np.array(masses)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # a0 (open water) stem, plotted separately so it can be orange
    markerline0, stemlines0, _ = ax.stem(
        locs[:1], masses[:1], basefmt=" ", label="open water ($a_0$)"
    )
    plt.setp(markerline0, color="C1", markersize=7)
    plt.setp(stemlines0, color="C1", linewidth=1.6)

    # remaining categories, unchanged blue
    markerline, stemlines, _ = ax.stem(
        locs[1:], masses[1:], basefmt=" ", label="extended state"
    )
    plt.setp(markerline, color="C0", markersize=7)
    plt.setp(stemlines, color="C0", linewidth=1.6)
    
    for hb in h_bnds:
        ax.axvline(hb, color="gray", linestyle="--", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ice thickness $h$ (m)")
    ax.set_ylabel("Mass")
    ax.set_title("Extended state: point masses")

    return ax, locs, masses, labels

def plot_extended_state_bars(x, h_bnds, ax=None, width=None):
    """Plot the extended state vector as paired bar plots (x_l on the left,
    x_u on the right of each category)."""
    
    x = np.asarray(x, dtype=float)
    h_bnds = np.asarray(h_bnds, dtype=float)

    I = (len(x) - 1) // 2
    if len(x) != 1 + 2 * I:
        raise ValueError("x must have length 1 + 2*I")
    if len(h_bnds) != I + 1:
        raise ValueError("h_bnds must have I+1 boundaries for I categories")

    a0 = x[0]
    x_l = x[1::2]
    x_u = x[2::2]

    if width is None:
        span = h_bnds[-1] - h_bnds[0]
        width = 0.02 * span   # narrow bars by default

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # a0 bar, centered at h=0
    ax.bar(0.0, a0, width=width, color="C1")

    # x_l bar just inside the left edge of each category,
    # x_u bar just inside the right edge
    for i in range(I):
        Hl, Hu = h_bnds[i], h_bnds[i + 1]
        ax.bar(Hl + width, x_l[i], width=width, color="C0")
        ax.bar(Hu - width, x_u[i], width=width, color="C0")

    for hb in h_bnds:
        ax.axvline(hb, color="gray", linestyle="--", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ice thickness $h$ (m)")
    ax.set_ylabel("Mass")
    ax.set_title("Extended state: category bars")

    return ax


def build_H_tilde(h_bnds):
    """Build the length-2I vector of thickness locations for x_i^l, x_i^u."""
    h_bnds = np.asarray(h_bnds, dtype=float)
    I = len(h_bnds) - 1
    H_tilde = np.empty(2 * I)
    H_tilde[0::2] = h_bnds[:-1]
    H_tilde[1::2] = h_bnds[1:]
    return H_tilde


def optimal_state_L2(x_f, h_bnds, Hbar_a, solver=None, verbose=False):
    """
    Solve for the analysis extended state x^a closest to the forecast
    x^f (in squared L2 norm) subject to the sum-to-one constraint and
    the analysis mean-thickness constraint:

        x^a = argmin_x ||x - x_f||^2
        s.t. sum_i x_i = 1
             sum_{i=1}^{2I} (H_tilde_i - Hbar_a) x_i = 0
             x_i >= 0

    Parameters
    ----------
    x_f : array_like, shape (1 + 2*I,)
        Forecast extended state, e.g. from extended_state().
    h_bnds : array_like, shape (I+1,)
        Thickness category boundaries.
    Hbar_a : float
        Analysis mean sea ice thickness (the target Hbar^a).
    solver : str, optional
        cvxpy solver name (e.g. "OSQP", "ECOS"). Defaults to cvxpy's
        automatic choice.
    verbose : bool
        Passed to cvxpy's solve() for solver diagnostics.

    Returns
    -------
    x_a : ndarray, shape (1 + 2*I,)
        Analysis extended state.
    problem : cvxpy.Problem
        The solved problem object (problem.status, problem.value, etc.
        useful for diagnostics).
    """
    x_f = np.asarray(x_f, dtype=float)
    h_bnds = np.asarray(h_bnds, dtype=float)

    I = (len(x_f) - 1) // 2
    if len(x_f) != 1 + 2 * I:
        raise ValueError("x_f must have length 1 + 2*I")

    H_tilde = build_H_tilde(h_bnds)   # length 2I, excludes a0

    n = 1 + 2 * I

    # create the n-dimensional optimization variable (for 5 ice thickness categories n = 11)
    x = cp.Variable(n)

    # Set up the objective, i.e want you want to minimize with respect to
    objective = cp.Minimize(cp.sum_squares(x - x_f))

    # Set the constraints
    constraints = [
        cp.sum(x) == 1,
        (H_tilde - Hbar_a) @ x[1:] == 0,   # excludes x[0] = a0
        x >= 0,
    ]

    # define the problem, it is immutable, it cannot be changed once it is created
    problem = cp.Problem(objective, constraints)

    # When solver is set to NONE, cvxpy auto-selects based on the problem
    # Since our problem is QP, it is likely using OSQP
    problem.solve(solver=solver, eps_abs=1e-7, eps_rel=1e-7, max_iter=500000, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: status = {problem.status}")

    return x.value, problem

def estimate_Bhat(X):
    """
    Estimate the ensemble covariance matrix Bhat 
    from an ensemble of extended state vectors.

    Parameters
    ----------
    X : ndarray, shape (N, n)
        Ensemble of N extended state vectors, each of length n = 1 + 2*I.

    Returns
    -------
    Bhat : ndarray, shape (n, n)
        Sample covariance matrix of the ensemble.

    """
    X = np.asarray(X, dtype=float)     # Matrix (Nens x Nvar)
    N_ens, n_var = X.shape         
    Xmean = X.mean(axis=0)             # vector (Nvar x 1)
    D = X-Xmean                        # Matrix (Nens x Nvar)
    Bhat= (D.T @ D)/(N_ens -1)         # Matrix (Nvar x Nvar)    

    return Bhat

def optimal_state_Binv(x_f, h_bnds, Hbar_a, Binv, solver=None, verbose=False):
    """
    Solve for the analysis extended state x^a closest to the forecast x^f in the 
    Binv-weighted norm, subject to the sum-to-one and the analysis mean-thickness 
    constraint.
        x^a = argmin_x (x-x^f)^T Binv (x-x^f)

    Parameters
    -------------

    x_f : array_like, shape (2*I + 1, ). 
        Forecast extended state, single ensemble member
    h_bnds : array_like, shape (I+1, ).
        Thickness category bounds
    Hbar_a : float
        Analysis mean sea ice thickness from step one
    Binv : ndarray, shape (2*I + 1, 2*I + 1).
        Psuedo inverse of the ensemble covariance
    Solver : str, optional
        cvxpy solver name
    Verbose: bool
        Passed to cvxpy's solve()

    Returns
    -------------
    x_a : ndarray, shape (2I+1, ).
        Analysis extended state
    problem : cvxpy. Problem
    
    """
    # Get number of ice categories from x_f
    I = (len(x_f) - 1) // 2

    # Make sure Binv is correct size 
    if Binv.shape != (2*I+1, 2*I +1): 
        raise ValueError(f"Binv must have shape ({2*I+1},{2*I+1}), got {Binv.shape}")
    
    x_a = cp.Variable(2*I+1)
    H_tilde = build_H_tilde(h_bnds)   
    
    
    #objective = cp.Minimize(cp.quad_form(x_a-x_f,Binv))
    objective = cp.Minimize(cp.quad_form(x_a - x_f, cp.psd_wrap(Binv)))
    constraints = [
        x_a >= 0, 
        cp.sum(x_a) == 1,
        (Hbar_a-H_tilde)@x_a[1:] == 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: status = {problem.status}")

    return x_a.value, problem


