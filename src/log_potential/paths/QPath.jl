mutable struct QPath{T<:Real} <: ParametrizedPath{T}
    t::T
	log_potential::Function
    prep
    backend::AbstractADType
end

# These are just to keep track of things if we come up with some better parametrization...
function q_to_param(q::T) where {T <: Real}
    return logit(q)
end

function param_to_q(param)
    return logistic(param)
end

function QPath(q0::T, backend::AbstractADType) where {T <: Real}
    t0 = q_to_param(q0)
    function __log_potential(t, log_potentials::AbstractVector{Float64}, β)
        V0, V1 = log_potentials
        q = param_to_q(t) 
        p = 1 - q
        return logweightaddexp(1 - β, p * V0, β, p * V1) / p
    end
    prep = prepare_path_gradient(__log_potential, t0, backend)
    return QPath(t0, __log_potential, prep, backend) 
end

function gradient(path::QPath, log_potentials::AbstractVector{Float64}, β)
    return path_gradient(path.log_potential, path.prep, path.t, log_potentials, β, path.backend)
end

function log_potential(path::QPath, log_potentials::AbstractVector{Float64}, β)
    return path.log_potential(path.t, log_potentials, β)
end

extract_param(path::QPath) = path.t
extract_reparam(path::QPath) = param_to_q(path.t)

function set_param!(path::QPath, t::T) where {T <: Real}
    path.t = t
end