
# Generalization of q-path where $p = 1 - q$ is allowed for any value in [-\infty, 1)
mutable struct PowerMeansPath{T<:Real} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

# These are just to keep track of things if we come up with some better parametrization...
function p_to_param(p::T) where {T <: Real}
    return log(1 - p)
end

function param_to_p(param)
    return 1 - exp(param)
end

function PowerMeansPath(p0::T, backend::AbstractADType) where {T <: Real}
    t0 = p_to_param(p0)
    return PowerMeansPath(t0, nothing, backend)
end

(path::PowerMeansPath)(t, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    p = param_to_p(t)
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    elseif p == 0.0
        return (1 - β) * V0 + β * V1
    elseif V1 == -Inf
        return log(1 - β) / p + V0
    else
        return V0 + logweightaddexp(1 - β, 1.0, β, p * (V1 - V0)) / p
    end
end

function log_potential(path::PowerMeansPath, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

extract_param(path::PowerMeansPath) = path.t
extract_reparam(path::PowerMeansPath) = param_to_q(path.t)

function set_param!(path::PowerMeansPath, t::T) where {T <: Real}
    path.t = t
end
