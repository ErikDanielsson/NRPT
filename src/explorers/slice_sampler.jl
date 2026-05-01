mutable struct SliceSampler <: Explorer
    w::Float64
    p::Int
end

SliceSampler() = SliceSampler(10., 3)

# 1D step: x is a scalar, lp is a Float64 -> Float64 log-potential closure
@inline function _step_coord(
    explorer::SliceSampler,
    problem::PathProblem,
    x::Vector{Float64},
    β::Float64,
    coord_ref::Ref,
    lp_buff::LP
) where {LP <: AbstractVector{Float64}}
    z = log_potential!(problem, x, β, lp_buff) - rand(Exponential())
    if z == -Inf
        error("Slice sampler is outside support at point $x at β=$β, [V0, V1] = $(lp_buff)")
    elseif isnan(z)
        error("NaN in slice sampler at $x")
    end

    start_val = coord_ref[]

    U = rand()
    L = start_val - U * explorer.w
    R = L + explorer.w
    K = explorer.p

    @inbounds coord_ref[] = L
    lp_L = log_potential!(problem, x, β, lp_buff)
    @inbounds coord_ref[] = R
    lp_R = log_potential!(problem, x, β, lp_buff)

    for _ in 1:K 
        if !(z < lp_L || z < lp_R)
            break
        end
        V = rand()
        if V < 0.5
            L = 2L - R
            coord_ref[] = L
            lp_L = log_potential!(problem, x, β, lp_buff)
        else
            R = 2R - L
            coord_ref[] = R
            lp_R = log_potential!(problem, x, β, lp_buff)
        end
    end
    coord_ref[] = start_val
    return _shrink_1d(explorer, problem, L, R, z, x, β, coord_ref, lp_buff)
end

function _shrink_1d(
    explorer::SliceSampler,
    problem::PathProblem,
    L::Float64,
    R::Float64,
    z::Float64,
    x::Vector{Float64},
    β::Float64,
    coord_ref::Ref,
    lp_buff::LP
) where {LP <: AbstractVector{Float64}}
    start_val = coord_ref[]
    L_bar = L
    R_bar = R
    y = L_bar + rand() * (R_bar - L_bar)
    coord_ref[] = y
    lp_y = log_potential!(problem, x, β, lp_buff)

    i = 0
    while true
        i += 1
        coord_ref[] = start_val
        if z <= lp_y && _accept_1d(explorer, problem, β, L, R, z, x, y, coord_ref, lp_buff)
            coord_ref[] = y
            break
        end
        if y < coord_ref[]
            L_bar = y
        else
            R_bar = y
        end
        y = L_bar + rand() * (R_bar - L_bar)
        coord_ref[] = y
        lp_y = log_potential!(problem, x, β, lp_buff)
        if i > 1000
            error("Over a thousand shrinks: $L_bar, $R_bar, $y, $z")
        end
    end
    return y
end

function _accept_1d(
    explorer::SliceSampler,
    problem::PathProblem,
    β::Float64,
    L::Float64,
    R::Float64,
    z::Float64,
    x::Vector{Float64},
    y::Float64,
    coord_ref,
    lp_buff::Vector{Float64},
)
    start_val = coord_ref[]
    L_hat = L
    R_hat = R
    coord_ref[] = L_hat
    lp_L_hat = log_potential!(problem, x, β, lp_buff)
    coord_ref[] = R_hat
    lp_R_hat = log_potential!(problem, x, β, lp_buff)
    D = false
    while (R_hat - L_hat) > 1.1 * explorer.w
        M = (R_hat + L_hat) / 2.0
        if (start_val < M && y >= M) || (start_val >= M && y < M)
            D = true
        end
        if y < M
            R_hat = M
            coord_ref[] = R_hat
            lp_R_hat = log_potential!(problem, x, β, lp_buff)
        else
            L_hat = M
            coord_ref[] = L_hat
            lp_L_hat = log_potential!(problem, x, β, lp_buff)
        end
        
        if D && z >= lp_L_hat && z >= lp_R_hat
            coord_ref[] = start_val
            return false
        end
    end
    coord_ref[] = start_val
    return true
end

# # Scalar dispatch
# function step(explorer::SliceSampler, problem::PathProblem, x::Float64, β)
#     lp = t -> log_potential(problem, t, β)
#     return _step_1d(explorer, lp, x)
# end

function step!(explorer::SliceSampler, problem::PathProblem, x::Float64, β::Float64, lp_buff::LP) where {LP <: AbstractVector{Float64}}
    lp(t::Float64) = log_potential!(problem, t, β, lp_buff)
    _step_1d(explorer, lp, x)
end

# # Multivariate dispatch: coordinate-wise slice sampling
# function step(explorer::SliceSampler, problem::PathProblem, x::Vector{Float64}, β)
#     y = copy(x)
#     for i in eachindex(y)
#         xi = y[i]
#         lp = t::Float64 -> (y[i] = t; v = log_potential(problem, y, β); y[i] = xi; v)
#         y[i] = _step_1d(explorer, lp, xi)
#     end
#     return y
# end

function step!(explorer::SliceSampler, problem::PathProblem, x::Vector{Float64}, β::Float64, lp_buff::LP) where {LP <: AbstractVector{Float64}}
    for i in eachindex(x)
        coord_ref = Ref(x, i)
        _step_coord(explorer, problem, x, β, coord_ref, lp_buff)
    end
end

function step(explorer::SliceSampler, problem::PathProblem, x::Vector{Float64}, β::Float64, lp_buff::LP) where {LP <: AbstractVector{Float64}}
    x = copy(x)
    step!(explorer, problem, x, β, lp_buff)
    return x
end

function step(explorer::SliceSampler, problem::PathProblem, x::Float64, β::Float64, lp_buff::LP) where {LP <: AbstractVector{Float64}}
    return step!(explorer, problem, x, β, lp_buff)
end