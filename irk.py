import numpy as np
import mpmath as mp
mp.dps = 200

from generation.gauss_legendre import build_gauss_legendre_irk
from generation.radau import build_radau_irk
from generation.lobatto import build_lobatto_IIIC_irk

#defines the finite difference Jacobian
def finite_diff_jac(fun, x, eps=1e-8):
    n = len(x)
    f0 = fun(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    return J

#solves a nonlinear system with Newton's method
def newton_solve(residual, y0, jac=None, tol=1e-10, max_iter=12):
    y = y0.copy()
    for it in range(1, max_iter + 1):
        r = residual(y)
        nr = np.linalg.norm(r)
        if nr < tol: return y, True
        J = jac(y) if jac else finite_diff_jac(residual, y)
        try: dy = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError: return y, False
        y += dy
        if np.linalg.norm(dy) < tol: return y, True
    return y, False

#loads the Butcher tableau
def get_tableau(family, s):
    family = family.lower()
    if family == "gauss":
        A, b, c = build_gauss_legendre_irk(s)
    elif family == "radau":
        A, b, c = build_radau_irk(s)
    elif family == "lobatto":
        A, b, c = build_lobatto_IIIC_irk(s)
    else:
        raise ValueError("Unknown IRK family")

    A = np.array([[float(A[i][j]) for j in range(s)] for i in range(s)])
    b = np.array([float(b[i]) for i in range(s)])
    c = np.array([float(c[i]) for i in range(s)])
    return A, b, c

#defines the step for the collocation method
def step_collocation(f, t, y, h, A, b, c, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8):
    s = len(b)
    n = len(y)
    Y = np.tile(y, (s, 1))
    t_nodes = t + c * h

    #builds the residual
    def residual(z_flat):
        Z = z_flat.reshape(s, n)
        R = np.zeros_like(Z)
        for i in range(s):
            acc = np.zeros(n)
            for j in range(s): acc += A[i, j] * f(t_nodes[j], Z[j])
            R[i] = Z[i] - y - h * acc
        return R.ravel()

    #builds the Jacobian
    def jacobian(z_flat):
        Z = z_flat.reshape(s, n)
        J_full = np.zeros((s * n, s * n))
        for j in range(s):
            Jf = jac(t_nodes[j], Z[j]) if jac else finite_diff_jac(lambda z: f(t_nodes[j], z), Z[j], eps=fd_eps)
            for i in range(s):
                block = -h * A[i, j] * Jf
                if i == j: block += np.eye(n)
                J_full[i*n:(i+1)*n, j*n:(j+1)*n] = block
        return J_full

    f0 = f(t, y)
    Y = np.array([y + (c[i] * h) * f0 for i in range(s)], dtype=float)
    z0 = Y.ravel()
    z_star, ok = newton_solve(residual, z0, jacobian, tol=tol, max_iter=max_iter)
    if not ok: return None, False
    Y = z_star.reshape(s, n)
    K = np.array([f(t_nodes[i], Y[i]) for i in range(s)])
    y_next = y + h * np.sum(b[:, None] * K, axis=0)
    return y_next, True

#defines the WRMS norm
def wrms_norm(err, y, y_new, atol=1e-6, rtol=1e-3):
    y = np.asarray(y, float)
    y_new = np.asarray(y_new, float)
    err = np.asarray(err, float)
    d = len(y)
    atol = np.full(d, atol) if np.ndim(atol) == 0 else np.asarray(atol)
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    scale = np.maximum(scale, 1e-30)
    return float(np.sqrt(np.mean((err / scale) ** 2)))

#defines the Euler defect PI controller
def euler_defect_pi_controller(y_n, y_np1, f_n, h, atol=1e-6, rtol=1e-3, E_prev=None, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e1):
    y_euler = y_n + h * f_n
    defect = y_np1 - y_euler
    E = wrms_norm(defect, y_n, y_np1, atol, rtol)

    #calculates the adaptive factor
    if E_prev is None: factor = safety * (1.0 / max(E, 1e-16)) ** kP
    else: factor = safety * (1.0 / max(E, 1e-16)) ** kP * (E_prev / max(E, 1e-16)) ** kI

    #clips the factor to the growth and shrink factors
    factor = np.clip(factor, shrink, growth)
    h_new = float(np.clip(h * factor, h_min, h_max))
    return h_new, E

#adaptive IRK solver
def solve_collocation_adaptive(f, t_span, y0, h0, family="gauss", s=3, jac=None, tol=1e-10, max_iter=12, atol=1e-6, rtol=1e-3, max_steps=10000, max_rejects=20, E_max=10.0, h_min=1e-12, h_max=1e1, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2):
    A, b, c = get_tableau(family, s)
    t0, tf = t_span
    t = t0
    y = np.asarray(y0, float)
    h = h0
    T = [t]
    Y = [y.copy()]
    E_hist = []

    #main loop for the adaptive step size solver
    while t < tf and len(T) < max_steps:
        h = min(h, tf - t)
        if h < h_min: break
        rejects = 0

        #retry loop
        while True:
            f_n = f(t, y)

            #attempts a single FIRK step
            y_next, newton_ok = step_collocation(f, t, y, h, A, b, c, jac=jac, tol=tol, max_iter=max_iter)

            #rejects on Newton failure
            if not newton_ok:
                h = max(shrink * h, h_min)
                rejects += 1
                if rejects > max_rejects: raise RuntimeError(f"FIRK Newton failed repeatedly at t={t:.6e}")
                continue

            #computes Euler-defect step size and error
            h_new, E = euler_defect_pi_controller(y, y_next, f_n, h, atol, rtol, E_hist[-1] if E_hist else None, safety, kP, kI, growth, shrink, h_min, h_max)

            #rejects the step if the defect is catastrophically large
            if E > E_max:
                h = max(shrink * h, h_min)
                rejects += 1
                if rejects > max_rejects: raise RuntimeError(f"FIRK adaptive solver failed: E={E:.2e} too large at t={t:.6e}")
                continue

            #accepts the step and breaks the retry loop
            break

        #updates the state after successful step
        t += h
        y = y_next
        T.append(t)
        Y.append(y.copy())
        E_hist.append(E)

        #updates the step size after successful step
        h = h_new
    return np.array(T), np.array(Y), np.array(E_hist)