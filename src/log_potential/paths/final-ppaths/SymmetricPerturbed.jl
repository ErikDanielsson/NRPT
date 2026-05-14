mutable struct SymmetricPerturbed{T <: AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

SymmetricPerturbed(backend::AbstractADType=AutoForwardDiff()) = SymmetricPerturbed([-10., -10.], nothing, backend)

(path::SymmetricPerturbed)(t, log_potentials::LP, β) where {LP <: AbstractVector{<:Real}} = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        forward = logistic(t[1]) * softplus(V0 - V1) + logistic(t[2]) * softplus(V1 - V0)
        return V0 * (1 - β) + β * V1 + (1 - β) * β * forward
    end
end

function log_potential(path::SymmetricPerturbed, log_potentials::LP, β) where {LP <: AbstractVector{<:Real}}
    return path(path.t, log_potentials, β)
end

extract_param(path::SymmetricPerturbed) = path.t

function extract_reparam(path::SymmetricPerturbed)
    t = path.t
    return logistic.(t)
end

function set_param!(path::SymmetricPerturbed, t::T) where {T <: AbstractVector}
    return path.t = t
end
