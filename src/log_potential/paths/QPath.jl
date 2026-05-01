mutable struct QPath{T<:Real} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

# These are just to keep track of things if we come up with some better parametrization...
function q_to_param(p::T) where {T <: Real}
    return logit(p)
end

function param_to_p(param)
    return logistic(param)
end

function QPath(p0::T, backend::AbstractADType) where {T <: Real}
    t0 = q_to_param(p0)
    return QPath(t0, nothing, backend)
end

(path::QPath)(t, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    p = param_to_p(t)
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    elseif V1 == -Inf
        return log(1 - β) / p + V0
    else
        return V0 + logweightaddexp(1 - β, 0.0, β, p * (V1 - V0)) / p
    end
end

function log_potential(path::QPath, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

extract_param(path::QPath) = path.t
extract_reparam(path::QPath) = param_to_p(path.t)

function set_param!(path::QPath, t::T) where {T <: Real}
    path.t = t
end
