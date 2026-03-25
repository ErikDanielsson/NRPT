mutable struct SliceSampler <: Explorer
    w::Float64
    p::Int
end

SliceSampler() = SliceSampler(10., 3)

# 1D step: x is a scalar, lp is a Float64 -> Float64 log-potential closure
function _step_1d(explorer::SliceSampler, lp, x::Float64)
    z = lp(x) - rand(Exponential())
    if z == -Inf
        error("Slice sampler is outside support at point $x")
    elseif isnan(z)
        error("NaN in slice sampler at $x")
    end

    U = rand()
    L = x - U * explorer.w
    R = L + explorer.w
    K = explorer.p
    while K > 0 && (z < lp(L) || z < lp(R))
        V = rand()
        if V < 0.5
            L = 2L - R
        else
            R = 2R - L
        end
        K -= 1
    end
    return _shrink_1d(explorer, lp, L, R, z, x)
end

function _shrink_1d(explorer::SliceSampler, lp, L::Float64, R::Float64, z::Float64, x::Float64)
    L_bar = L
    R_bar = R
    y = L_bar + rand() * (R_bar - L_bar)
    i = 0
    while true
        i += 1
        if z <= lp(y) && _accept_1d(explorer, lp, L, R, z, x, y)
            break
        end
        if y < x
            L_bar = y
        else
            R_bar = y
        end
        y = L_bar + rand() * (R_bar - L_bar)
        if i > 1000
            println(L_bar)
            println(R_bar)
            println(y)
            println(z)
        end
    end
    return y
end

function _accept_1d(explorer::SliceSampler, lp, L::Float64, R::Float64, z::Float64, x::Float64, y::Float64)
    L_hat = L
    R_hat = R
    D = false
    while (R_hat - L_hat) > 1.1 * explorer.w
        M = (R_hat + L_hat) / 2.0
        if (x < M && y >= M) || (x >= M && y < M)
            D = true
        end
        if y < M
            R_hat = M
        else
            L_hat = M
        end
        if D && z >= lp(L_hat) && z >= lp(R_hat)
            return false
        end
    end
    return true
end

# Scalar dispatch
function step(explorer::SliceSampler, problem::PathProblem, x::Float64, β)
    lp = t -> log_potential(problem, t, β)
    return _step_1d(explorer, lp, x)
end

# Multivariate dispatch: coordinate-wise slice sampling
function step(explorer::SliceSampler, problem::PathProblem, x::Vector{Float64}, β)
    y = copy(x)
    for i in eachindex(y)
        xi = y[i]
        lp = t -> (y[i] = t; v = log_potential(problem, y, β); y[i] = xi; v)
        y[i] = _step_1d(explorer, lp, xi)
    end
    return y
end
