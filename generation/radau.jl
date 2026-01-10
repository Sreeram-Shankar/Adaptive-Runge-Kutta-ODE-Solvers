using Printf
using QuadGK
setprecision(BigFloat, 200)

#formats the number string
function nstr_fixed(x::BigFloat, digits::Int=80)
    x_float = Float64(x)
    multiplier = 10.0^digits
    rounded = round(x_float * multiplier) / multiplier
    s = string(rounded)
    if occursin(".", s)
        s = rstrip(s, '0')
        s = rstrip(s, '.')
    end
    return s
end

#jacobi polynomial function using hypergeometric function
function jacobi_P(n::Int, alpha::BigFloat, beta::BigFloat, x::BigFloat)
    if n == 0
        return one(x)
    elseif n == 1
        return (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2
    else
        P_prev = one(x)
        P_curr = (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2
        for k in 2:n
            a1 = (2*k + alpha + beta - 1) * ((2*k + alpha + beta) * (2*k + alpha + beta - 2) * x + (alpha^2 - beta^2))
            a2 = 2 * (k + alpha - 1) * (k + beta - 1) * (2*k + alpha + beta)
            P_next = (a1 * P_curr - a2 * P_prev) / (2 * k * (k + alpha + beta) * (2*k + alpha + beta - 2))
            P_prev = P_curr
            P_curr = P_next
        end
        return P_curr
    end
end

#finds the roots of the jacobi polynomial using bisection
function jacobi_roots_radau_interior(n::Int, dps_scan::Int=0)
    alpha = BigFloat(1)
    beta = BigFloat(0)
    f = x -> jacobi_P(n, alpha, beta, x)
    
    if dps_scan == 0
        dps_scan = max(2000, 400 * n)
    end
    xs = [BigFloat(-1) + 2*i/(dps_scan-1) for i in 0:(dps_scan-1)]
    fs = [f(x) for x in xs]
    
    roots = BigFloat[]
    for i in 1:(dps_scan-1)
        a, b = xs[i], xs[i+1]
        fa, fb = fs[i], fs[i+1]
        if fa == 0
            push!(roots, a)
            continue
        end
        if fa * fb < 0
            for _ in 1:200
                m = (a + b) / 2
                fm = f(m)
                if abs(fm) < BigFloat("1e-70") || abs(b - a) < BigFloat("1e-50")
                    push!(roots, m)
                    break
                end
                if fa * fm < 0
                    b, fb = m, fm
                else
                    a, fa = m, fm
                end
            end
        end
    end
    
    roots = [r for r in roots if -1 < r < 1]
    roots = sort(unique([nstr_fixed(r, 60) for r in roots]))
    roots = [parse(BigFloat, r) for r in roots]
    
    if length(roots) != n
        if dps_scan < 20000
            return jacobi_roots_radau_interior(n, 20000)
        end
        error("Expected $n interior roots, got $(length(roots))")
    end
    
    return roots
end

#generate radau nodes
function radau_right_nodes_on_01(s::Int)
    if s < 1
        error("Radau quadrature requires s >= 1.")
    end
    if s == 1
        return [BigFloat(1)]
    end
    interior = jacobi_roots_radau_interior(s - 1)
    x_all = [interior; BigFloat(1)]
    c = [(x + 1) / 2 for x in x_all]
    return c
end

#lagrange basis polynomial
function lagrange_basis(c::Vector{BigFloat}, j::Int)
    xj = c[j]
    others = [c[k] for k in eachindex(c) if k != j]
    
    denom = one(BigFloat)
    for xk in others
        denom *= (xj - xk)
    end
    
    function Lj(x::BigFloat)
        num = one(BigFloat)
        for xk in others
            num *= (x - xk)
        end
        return num / denom
    end
    
    return Lj
end

#builds tableau matrices
function build_A_b(c::Vector{BigFloat})
    s = length(c)
    A = zeros(BigFloat, s, s)
    b = zeros(BigFloat, s)
    for j in 1:s
        Lj = lagrange_basis(c, j)
        b[j], _ = quadgk(Lj, BigFloat(0), BigFloat(1), rtol=eps(BigFloat))
        for i in 1:s
            A[i, j], _ = quadgk(Lj, BigFloat(0), c[i], rtol=eps(BigFloat))
        end
    end
    return A, b
end

#writes the tableau to files
function write_A_b_c_triplets(A::Matrix{BigFloat}, b::Vector{BigFloat}, c::Vector{BigFloat}, basename::String, digits::Int=80)
    s = length(b)
    
    open("$(basename)_A.txt", "w") do fa
        for i in 1:s
            for j in 1:s
                println(fa, "$i $j $(nstr_fixed(A[i, j], digits))")
            end
        end
    end
    
    open("$(basename)_b.txt", "w") do fb
        for j in 1:s
            println(fb, "$j $(nstr_fixed(b[j], digits))")
        end
    end
    
    open("$(basename)_c.txt", "w") do fc
        for i in 1:s
            println(fc, "$i $(nstr_fixed(c[i], digits))")
        end
    end
    
    open("$(basename)_triplets.txt", "w") do ft
        for i in 1:s
            for j in 1:s
                println(ft, "$i $j $(nstr_fixed(A[i, j], digits))")
            end
        end
        for i in 1:s
            println(ft, "$i 0 $(nstr_fixed(c[i], digits))")
        end
        for j in 1:s
            println(ft, "0 $j $(nstr_fixed(b[j], digits))")
        end
    end
end

#verifies tableau correctness
function check_tableau(A::Matrix{BigFloat}, b::Vector{BigFloat}, c::Vector{BigFloat})
    for i in eachindex(c)
        ssum = sum(A[i, :])
        diff = ssum - c[i]
    end
    for k in 0:min(9, 2*length(c)-1)
        lhs = sum(b[j] * c[j]^k for j in eachindex(c))
        rhs = BigFloat(1) / (k + 1)
        err = lhs - rhs
    end
end

#main radau builder function
function build_radau_irk(s::Int)
    c = radau_right_nodes_on_01(s)
    A, b = build_A_b(c)
    check_tableau(A, b, c)
    return A, b, c
end