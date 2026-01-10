using LinearAlgebra
using Statistics
setprecision(BigFloat, 200)
include("generation/gauss_legendre.jl")
include("generation/radau.jl")
include("generation/lobatto.jl")

#defines the finite difference Jacobian
function finite_diff_jac(fun, x, eps=1e-8)
    n = length(x)
    f0 = fun(x)
    J = zeros(n, n)
    for j in 1:n
        dx = zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    end
    return J
end

#solves a nonlinear system with Newton's method
function newton_solve(residual, y0, jac=nothing; tol=1e-10, max_iter=12)
    y = copy(y0)
    for _ in 1:max_iter
        r = residual(y)
        nr = norm(r)
        if nr < tol
            return y, true
        end
        J = jac !== nothing ? jac(y) : finite_diff_jac(residual, y)
        dy = try
            J \ (-r)
        catch _
            return y, false
        end
        y += dy
        if norm(dy) < tol
            return y, true
        end
    end
    return y, false
end

#loads the Butcher tableau from the generators
function get_tableau(family, s)
    family = lowercase(family)
    if family == "gauss"
        A, b, c = build_gauss_legendre_irk(s)
    elseif family == "radau"
        A, b, c = build_radau_irk(s)
    elseif family == "lobatto"
        A, b, c = build_lobatto_IIIC_irk(s)
    else
        error("Unknown family '$family', must be 'gauss', 'radau', or 'lobatto'.")
    end

    #converts the tableau to numpy arrays
    A = [[Float64(A[i, j]) for j in 1:s] for i in 1:s]
    b = [Float64(b[i]) for i in 1:s]
    c = [Float64(c[i]) for i in 1:s]
    return A, b, c
end


#defines the step for the collocation method
function step_collocation(f, t, y, h, A, b, c, jac=nothing; tol=1e-10, max_iter=12, fd_eps=1e-8)
    s = length(b)
    n = length(y)
    Y = repeat(reshape(y, 1, length(y)), s, 1)
    t_nodes = [t + c[i] * h for i in 1:s]

    #builds the residual
    function residual(z_flat)
        Z = reshape(z_flat, s, n)
        R = zeros(size(Z))
        for i in 1:s
            acc = zeros(n)
            for j in 1:s
                acc += A[i][j] * f(t_nodes[j], Z[j, :])
            end
            R[i, :] = Z[i, :] - y - h * acc
        end
        return vec(R)
    end

    #builds the Jacobian
    function jacobian(z_flat)
        Z = reshape(z_flat, s, n)
        J_full = zeros(s * n, s * n)
        for j in 1:s
            Jf_j = jac !== nothing ? jac(t_nodes[j], Z[j, :]) : finite_diff_jac(z -> f(t_nodes[j], z), Z[j, :], fd_eps)
            for i in 1:s
                block = -h * A[i][j] * Jf_j
                if i == j
                    block = block + I(n)
                end
                row = (i-1)*n+1:i*n
                col = (j-1)*n+1:j*n
                J_full[row, col] = block
            end
        end
        return J_full
    end

    z0 = vec(Y)
    z_star, ok = newton_solve(residual, z0, jacobian; tol=tol, max_iter=max_iter)
    if !ok
        return nothing, false
    end
    Y = reshape(z_star, s, n)
    K = zeros(s, n)
    for i in 1:s
        K[i, :] = f(t_nodes[i], Y[i, :])
    end
    y_next = y + h * sum([b[i] * K[i, :] for i in 1:s])
    return y_next, true
end


#main solver for any collocation method
function solve_collocation(f, t_span, y0, h, family="gauss", s=3, jac=nothing; tol=1e-10, max_iter=12, fd_eps=1e-8)
    A, b, c = get_tableau(family, s)
    t0, tf = t_span
    N = Int(ceil((tf - t0)/h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N+1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        y_next, ok = step_collocation(f, t_grid[n], Y[n, :], h, A, b, c, jac; tol=tol, max_iter=max_iter, fd_eps=fd_eps)
        if !ok
            error("Collocation Newton failed at step $n")
        end
        Y[n+1, :] = y_next
    end
    return collect(t_grid), Y
end

#defines the WRMS norm
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

#defines the Euler defect PI controller
function euler_defect_pi_controller(y_n, y_np1, f_n, h; atol=1e-6, rtol=1e-3, E_prev=nothing, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2, h_min=1e-12, h_max=1e1)
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

#adaptive IRK solver
function solve_collocation_adaptive(f, t_span, y0, h0; family="gauss", s=3, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8, atol=1e-6, rtol=1e-3, max_steps=10000, max_rejects=20, E_max=10.0, h_min=1e-12, h_max=1e1, safety=0.9, kP=0.5, kI=0.05, growth=2.0, shrink=0.2)
    A, b, c = get_tableau(family, s)
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
        y_next = nothing
        E = 0.0
        h_new = h

        #retry loop
        while true
            f_n = f(t, y)

            #attempts a single FIRK step
            y_next, newton_ok = step_collocation(f, t, y, h, A, b, c, jac; tol=tol, max_iter=max_iter, fd_eps=fd_eps)

            #rejects on Newton failure
            if !newton_ok
                h = max(shrink * h, h_min)
                rejects += 1
                if rejects > max_rejects
                    error("FIRK Newton failed repeatedly at t=$(t)")
                end
                continue
            end

            #computes Euler-defect step size and error
            h_new, E = euler_defect_pi_controller(y, y_next, f_n, h; atol=atol, rtol=rtol, E_prev=isempty(E_hist) ? nothing : E_hist[end], safety=safety, kP=kP, kI=kI, growth=growth, shrink=shrink, h_min=h_min, h_max=h_max)

            #rejects the step if the defect is catastrophically large
            if E > E_max
                h = max(shrink * h, h_min)
                rejects += 1
                if rejects > max_rejects
                    error("FIRK adaptive solver failed: E=$(E) too large at t=$(t)")
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

        #updates the step size after successful step
        h = h_new
    end
    return T, Y, E_hist
end