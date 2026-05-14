mutable struct AsymmetricPerturbed{T <: AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

AsymmetricPerturbed(backend::AbstractADType = AutoForwardDiff()) = AsymmetricPerturbed([-10.0], nothing, backend)

(path::AsymmetricPerturbed)(t, log_potentials::LP, β) where {LP <: AbstractVector{<:Real}} = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        forward = logistic(t[1]) * softplus(V1 - V0)
        return V0 * (1 - β) + β * V1 + (1 - β) * β * forward
    end
end

function log_potential(path::AsymmetricPerturbed, log_potentials::LP, β) where {LP <: AbstractVector{<:Real}}
    return path(path.t, log_potentials, β)
end

extract_param(path::AsymmetricPerturbed) = path.t

function extract_reparam(path::AsymmetricPerturbed)
    t = path.t
    return exp(t[1])
end

function set_param!(path::AsymmetricPerturbed, t::T) where {T <: AbstractVector}
    return path.t = t
end
