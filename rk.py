import numpy as np

#defines the step for RK2
def step_rk2(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + (h / 2) * (k1 + k2)

#defines the step for RK3
def step_rk3(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + (h / 2) * k1)
    k3 = f(t + h, y + h * (-k1 + 2 * k2))
    return y + (h / 6) * (k1 + 4 * k2 + k3)

#defines the step for RK4
def step_rk4(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + (h / 2) * k1)
    k3 = f(t + h / 2, y + (h / 2) * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

#defines the step for RK5
def step_rk5(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h * 1/5, y + h * (1/5) * k1)
    k3 = f(t + h * 3/10, y + h * (3/40 * k1 + 9/40 * k2))
    k4 = f(t + h * 4/5, y + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3))
    k5 = f(t + h * 8/9, y + h * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4))
    k6 = f(t + h, y + h * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5))
    return y + h * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)

#generic explicit RK for a given A tableau and b coefficients
def _step_rk_explicit(f, t, y, h, A, b):
    s = len(b)
    K = [None] * s
    for i in range(s):
        yi = y.copy()
        for j in range(i):
            yi += h * A[i][j] * K[j]
        ti = t + h * sum(A[i])
        K[i] = f(ti, yi)
    y_next = y + h * sum(b[i] * K[i] for i in range(s))
    return y_next

#defines the step for RK6
def step_rk6(f, t, y, h):
    A = [
        [0.0],
        [1.0/12.0, 0.0],
        [0.0, 1.0/6.0, 0.0],
        [1.0/16.0, 0.0, 3.0/16.0, 0.0],
        [21.0/16.0, 0.0, -81.0/16.0, 9.0/2.0, 0.0],
        [1344688.0/250563.0, 0.0, -1709184.0/83521.0, 1365632.0/83521.0, -78208.0/250563.0, 0.0],
        [-559.0/384.0, 0.0, 6.0, -204.0/47.0, 14.0/39.0, -4913.0/78208.0, 0.0],
        [-625.0/224.0, 0.0, 12.0, -456.0/47.0, 48.0/91.0, 14739.0/136864.0, 6.0/7.0, 0.0],
    ]
    b = [7.0/90.0, 0.0, 0.0, 16.0/45.0, 16.0/45.0, 0.0, 2.0/15.0, 7.0/90.0]
    return _step_rk_explicit(f, t, y, h, A, b)

#defines the step for RK7
def step_rk7(f, t, y, h):
    A = [
        [0.0],
        [1.0/4.0, 0.0],
        [5.0/72.0, 1.0/72.0, 0.0],
        [1.0/32.0, 0.0, 3.0/32.0, 0.0],
        [106.0/125.0, 0.0, -408.0/125.0, 352.0/125.0, 0.0],
        [1.0/48.0, 0.0, 0.0, 8.0/33.0, 125.0/528.0, 0.0],
        [-1263.0/2401.0, 0.0, 0.0, 39936.0/26411.0, -64125.0/26411.0, 5520.0/2401.0, 0.0],
        [37.0/392.0, 0.0, 0.0, 0.0, 1625.0/9408.0, -2.0/15.0, 61.0/6720.0, 0.0],
        [17176.0/25515.0, 0.0, 0.0, -47104.0/25515.0, 1325.0/504.0, -41792.0/25515.0, 20237.0/145800.0, 4312.0/6075.0, 0.0],
        [-23834.0/180075.0, 0.0, 0.0, -77824.0/1980825.0, -636635.0/633864.0, 254048.0/300125.0, -183.0/7000.0, 8.0/11.0, -324.0/3773.0, 0.0],
        [12733.0/7600.0, 0.0, 0.0, -20032.0/5225.0, 456485.0/80256.0, -42599.0/7125.0, 339227.0/912000.0, -1029.0/4180.0, 1701.0/1408.0, 5145.0/2432.0, 0.0],
    ]
    b = [
        13.0/288.0, 0.0, 0.0, 0.0, 0.0, 32.0/125.0,
        31213.0/144000.0, 2401.0/12375.0, 1701.0/14080.0, 2401.0/19200.0, 19.0/450.0
    ]
    return _step_rk_explicit(f, t, y, h, A, b)

#defines the wrms norm
def wrms_norm(err, y, y_new, atol=1e-6, rtol=1e-3):
    y = np.asarray(y, float)
    y_new = np.asarray(y_new, float)
    err = np.asarray(err, float)
    d = len(y)
    atol = np.full(d, atol) if np.ndim(atol) == 0 else np.asarray(atol)
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    scale = np.maximum(scale, 1e-30)
    return float(np.sqrt(np.mean((err / scale) ** 2)))

#defines the Euler defect PI controller, although defect ~ h^2 so kP = 1/2
def euler_defect_pi_controller(y_n, y_np1, f_n, h, atol=1e-6, rtol=1e-3, E_prev=None, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e2):
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

#main solver for the RK adaptive step size solver
def solve_rk_adaptive(stepper, f, t_span, y0, h0, atol=1e-6, rtol=1e-3, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e1, max_steps=10_000, max_rejects=20, E_max=10.0):
    t0, tf = t_span
    t = t0
    y = np.asarray(y0, float); h = float(h0); T = [t]; Y = [y.copy()]; E_hist = []
    

    #main loop for the adaptive step size solver
    while t < tf and len(T) < max_steps:
        h = min(h, tf - t)
        if h < h_min: break
        rejects = 0

        #retry loop
        while True:
            f_n = f(t, y)

            #takes a single explicit RK step
            y_next = stepper(f, t, y, h)

            #computes Euler-defect step size and error
            h_new, E = euler_defect_pi_controller(y, y_next, f_n, h, atol, rtol, E_hist[-1] if E_hist else None, safety, kP, kI, growth, shrink, h_min, h_max)

            #rejects the step if the defect is catastrophically large
            if E > E_max:
                h = max(shrink * h, h_min)
                rejects += 1
                if rejects > max_rejects: raise RuntimeError(f"RK adaptive solver failed: E={E:.2e} too large at t={t:.6e}")
                continue
            #accepts the step and breaks the retry loop
            break

        #updates the state after successful step
        t += h
        y = y_next
        T.append(t)
        Y.append(y.copy())
        E_hist.append(E)
        h = h_new
    return np.array(T), np.array(Y), np.array(E_hist)

#defines the adaptive step size solvers for RK2-7
def solve_rk2_adaptive(f, t_span, y0, h0, **kw): return solve_rk_adaptive(step_rk2, f, t_span, y0, h0, **kw)
def solve_rk3_adaptive(f, t_span, y0, h0, **kw): return solve_rk_adaptive(step_rk3, f, t_span, y0, h0, **kw)
def solve_rk4_adaptive(f, t_span, y0, h0, **kw): return solve_rk_adaptive(step_rk4, f, t_span, y0, h0, **kw)
def solve_rk5_adaptive(f, t_span, y0, h0, **kw): return solve_rk_adaptive(step_rk5, f, t_span, y0, h0, **kw)
def solve_rk6_adaptive(f, t_span, y0, h0, **kw): return solve_rk_adaptive(step_rk6, f, t_span, y0, h0, **kw)
def solve_rk7_adaptive(f, t_span, y0, h0, **kw): return solve_rk_adaptive(step_rk7, f, t_span, y0, h0, **kw)