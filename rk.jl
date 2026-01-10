#defines the step for RK1
function step_rk1(f, t, y, h)
    k1 = f(t, y)
    return y + h * k1
end

#defines the wrms norm
function wrms_norm(err, y, y_new; atol=1e-6, rtol=1e-3)
    y = Array{Float64}(y)
    y_new = Array{Float64}(y_new)
    err = Array{Float64}(err)
    d = length(y)
    atol_vec = isa(atol, Number) ? fill(atol, d) : Array{Float64}(atol)
    scale = atol_vec .+ rtol .* max.(abs.(y), abs.(y_new))
    scale = max.(scale, fill(1e-30, d))
    return sqrt(mean((err ./ scale).^2))
end

#defines the Euler defect PI controller, although defect ~ h^2 so kP = 1/2
function euler_defect_pi_controller(y_n, y_np1, f_n, h; atol=1e-6, rtol=1e-3, E_prev=nothing, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e2)
    y_euler = y_n + h .* f_n
    defect = y_np1 - y_euler
    E = wrms_norm(defect, y_n, y_np1; atol=atol, rtol=rtol)

    #calculates the adaptive factor
    factor = if E_prev === nothing
        safety * (1.0 / max(E, 1e-16))^kP
    else
        safety * (1.0 / max(E, 1e-16))^kP * (E_prev / max(E, 1e-16))^kI
    end

    #clips the factor to the growth and shrink factors
    factor = clamp(factor, shrink, growth)
    h_new = clamp(h * factor, h_min, h_max)
    return h_new, E
end

#main solver for RK1
function solve_rk1(f, t_span, y0, h)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n + 1, :] = step_rk1(f, t_grid[n], Y[n, :], h)
    end
    return collect(t_grid), Y
end

#defines the step for RK2
function step_rk2(f, t, y, h)
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + (h / 2) * (k1 + k2)
end

#main solver for RK2
function solve_rk2(f, t_span, y0, h)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n + 1, :] = step_rk2(f, t_grid[n], Y[n, :], h)
    end
    return collect(t_grid), Y
end

#defines the step for RK3
function step_rk3(f, t, y, h)
    k1 = f(t, y)
    k2 = f(t + h / 2, y + (h / 2) * k1)
    k3 = f(t + h, y + h * (-k1 + 2 * k2))
    return y + (h / 6) * (k1 + 4 * k2 + k3)
end

#main solver for RK3
function solve_rk3(f, t_span, y0, h)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n + 1, :] = step_rk3(f, t_grid[n], Y[n, :], h)
    end
    return collect(t_grid), Y
end

#defines the step for RK4
function step_rk4(f, t, y, h)
    k1 = f(t, y)
    k2 = f(t + h / 2, y + (h / 2) * k1)
    k3 = f(t + h / 2, y + (h / 2) * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

#main solver for RK4
function solve_rk4(f, t_span, y0, h)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n + 1, :] = step_rk4(f, t_grid[n], Y[n, :], h)
    end
    return collect(t_grid), Y
end

#defines the step for RK5
function step_rk5(f, t, y, h)
    k1 = f(t, y)
    k2 = f(t + h * 1/5, y + h * (1/5) * k1)
    k3 = f(t + h * 3/10, y + h * (3/40 * k1 + 9/40 * k2))
    k4 = f(t + h * 4/5, y + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3))
    k5 = f(t + h * 8/9, y + h * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4))
    k6 = f(t + h, y + h * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5))
    return y + h * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
end

#generic explicit RK for a given A tableau and b coefficients
function _step_rk_explicit(f, t, y, h, A, b)
    s = length(b)
    K = Vector{typeof(y)}(undef, s)
    for i in 1:s
        yi = copy(y)
        for j in 1:(i-1)
            yi += h * A[i][j] * K[j]
        end
        ti = t + h * sum(A[i])
        K[i] = f(ti, yi)
    end
    y_next = y + h * sum([b[i] * K[i] for i in 1:s])
    return y_next
end


#defines the step for RK6
function step_rk6(f, t, y, h)
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
end


#defines the step for RK7
function step_rk7(f, t, y, h)
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
end

#main solver for RK5
function solve_rk5(f, t_span, y0, h)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n + 1, :] = step_rk5(f, t_grid[n], Y[n, :], h)
    end
    return collect(t_grid), Y
end

#main solver for RK6
function solve_rk6(f, t_span, y0, h)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n + 1, :] = step_rk6(f, t_grid[n], Y[n, :], h)
    end
    return collect(t_grid), Y
end

#main solver for RK7
function solve_rk7(f, t_span, y0, h)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N + 1)
    Y = zeros(N + 1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n + 1, :] = step_rk7(f, t_grid[n], Y[n, :], h)
    end
    return collect(t_grid), Y
end

#main solver for the RK adaptive step size solver
function solve_rk_adaptive(stepper, f, t_span, y0, h0; atol=1e-6, rtol=1e-3, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e1, max_steps=10_000, max_rejects=20, E_max=10.0)
    t0, tf = t_span
    t = t0
    y = Array{Float64}(y0)
    h = float(h0)
    T = [t]
    Y = [copy(y)]
    E_hist = Float64[]

    #main loop for the adaptive step size solver
    while t < tf && length(T) < max_steps
        h = min(h, tf - t)
        if h < h_min
            break
        end
        rejects = 0
        y_next = copy(y)
        E = 0.0
        h_new = h

        #retry loop
        while true
            f_n = f(t, y)

            #takes a single explicit RK step
            y_next = stepper(f, t, y, h)

            #computes Euler-defect step size and error
            h_new, E = euler_defect_pi_controller(y, y_next, f_n, h; atol=atol, rtol=rtol, E_prev=isempty(E_hist) ? nothing : E_hist[end], safety=safety, kP=kP, kI=kI, growth=growth, shrink=shrink, h_min=h_min, h_max=h_max)

            #rejects the step if the defect is catastrophically large
            if E > E_max
                h = max(shrink * h, h_min)
                rejects += 1
                if rejects > max_rejects
                    error("RK adaptive solver failed: E=$(E) too large at t=$(t)")
                end
                continue
            end
            #accepts the step and breaks the retry loop
            break
        end

        #updates the state after successful step
        t += h
        y = y_next
        push!(T, t)
        push!(Y, copy(y))
        push!(E_hist, E)
        h = h_new
    end
    return T, Y, E_hist
end

#defines the adaptive step size solvers for RK2-7
solve_rk2_adaptive(f, t_span, y0, h0; kw...) = solve_rk_adaptive(step_rk2, f, t_span, y0, h0; kw...)
solve_rk3_adaptive(f, t_span, y0, h0; kw...) = solve_rk_adaptive(step_rk3, f, t_span, y0, h0; kw...)
solve_rk4_adaptive(f, t_span, y0, h0; kw...) = solve_rk_adaptive(step_rk4, f, t_span, y0, h0; kw...)
solve_rk5_adaptive(f, t_span, y0, h0; kw...) = solve_rk_adaptive(step_rk5, f, t_span, y0, h0; kw...)
solve_rk6_adaptive(f, t_span, y0, h0; kw...) = solve_rk_adaptive(step_rk6, f, t_span, y0, h0; kw...)
solve_rk7_adaptive(f, t_span, y0, h0; kw...) = solve_rk_adaptive(step_rk7, f, t_span, y0, h0; kw...)