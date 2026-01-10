using Printf
using QuadGK
setprecision(BigFloat, 100)

#legendre polynomial evaluation using recurrence relation
function legendre_poly(n::Int, x::BigFloat)
    if n == 0
        return one(x)
    elseif n == 1
        return x
    else
        P_prev = one(x)
        P_curr = x
        for k in 2:n
            P_next = ((2*k - 1) * x * P_curr - (k - 1) * P_prev) / k
            P_prev = P_curr
            P_curr = P_next
        end
        return P_curr
    end
end

#derivative of legendre polynomial
function legendre_poly_prime(n::Int, x::BigFloat)
    den = x^2 - 1
    if abs(den) > BigFloat("1e-30")
        return n * (x * legendre_poly(n, x) - legendre_poly(n-1, x)) / den
    else
        h = BigFloat("1e-30")
        return (legendre_poly(n, x + h) - legendre_poly(n, x - h)) / (2*h)
    end
end

#lobatto roots and weights generator
function lobatto_roots_and_weights(s::Int)
    if s < 2
        error("Lobatto quadrature requires at least 2 stages")
    end
    n = s - 1
    
    xs = [BigFloat(-1)]
    Pn = x -> legendre_poly(n, x)
    
    function Pn_prime(x::BigFloat)
        den = x*x - 1
        if abs(den) > BigFloat("1e-30")
            return n * (x * legendre_poly(n, x) - legendre_poly(n-1, x)) / den
        else
            h = BigFloat("1e-30")
            return (legendre_poly(n, x + h) - legendre_poly(n, x - h)) / (2*h)
        end
    end
    
    for k in 1:(n-1)
        x0 = cos(BigFloat(π) * BigFloat(k) / BigFloat(n))
        x = x0
        for _ in 1:100
            fx = Pn_prime(x)
            h = BigFloat("1e-30")
            dfx = (Pn_prime(x + h) - Pn_prime(x - h)) / (2*h)
            if abs(dfx) < BigFloat("1e-50")
                break
            end
            x_new = x - fx / dfx
            if abs(x_new - x) < eps(BigFloat) * 100
                break
            end
            x = x_new
        end
        push!(xs, x)
    end
    
    push!(xs, BigFloat(1))
    
    ws = BigFloat[]
    for (i, x) in enumerate(xs)
        if i == 1 || i == s
            w = BigFloat(2) / (n*(n+1))
        else
            w = BigFloat(2) / (n*(n+1) * (Pn(x)^2))
        end
        push!(ws, w)
    end
    
    return xs, ws
end

#lagrange basis polynomial
function lagrange_basis(c::Vector{BigFloat}, j::Int)
    xj = c[j]
    others = [c[k] for k in 1:length(c) if k != j]
    
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

#builds lobatto iic irk tableau
function build_lobatto_IIIC_irk(s::Int)
    xs, ws = lobatto_roots_and_weights(s)
    c = [(x + 1) / 2 for x in xs]
    b = [w / 2 for w in ws]
    
    A = zeros(BigFloat, s, s)
    for j in 1:s
        Lj = lagrange_basis(c, j)
        for i in 1:s
            A[i, j], _ = quadgk(Lj, BigFloat(0), c[i], rtol=eps(BigFloat))
        end
    end
    
    for i in 1:s
        rs = sum(A[i, :])
        diff = rs - c[i]
    end
    
    for k in 0:min(9, 2*s-3)
        lhs = sum(b[j] * c[j]^k for j in 1:s)
        rhs = BigFloat(1) / (k + 1)
        err = lhs - rhs
    end
    
    return A, b, c
end

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