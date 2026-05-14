mutable struct AsymmetricTwoParamNegPerturbed{T <: AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

AsymmetricTwoParamNegPerturbed(backend::AbstractADType = AutoForwardDiff()) = AsymmetricTwoParamNegPerturbed([-10.0, 0.0], nothing, backend)

(path::AsymmetricTwoParamNegPerturbed)(t, log_potentials::LP, β) where {LP <: AbstractVector{<:Real}} = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        forward = logistic(t[1]) * logaddexp(t[2], V0 - V1)
        return V0 * (1 - β) + β * V1 + (1 - β) * β * forward
    end
end

function log_potential(path::AsymmetricTwoParamNegPerturbed, log_potentials::LP, β) where {LP <: AbstractVector{<:Real}}
    return path(path.t, log_potentials, β)
end

extract_param(path::AsymmetricTwoParamNegPerturbed) = path.t

function extract_reparam(path::AsymmetricTwoParamNegPerturbed)
    t = path.t
    return [logistic(t[1]), t[2]]
end

function set_param!(path::AsymmetricTwoParamNegPerturbed, t::T) where {T <: AbstractVector}
    return path.t = t
end
