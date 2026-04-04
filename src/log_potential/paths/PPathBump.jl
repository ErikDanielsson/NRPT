mutable struct PPathBump{T<:AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

function PPathBump{T}(t0::T, backend::AbstractADType) where {T<:AbstractVector{<:Real}}
    return PPathBump(t0, nothing, backend)
end

# Convenience constructor: default initialisation t = 0 (cᵢ = dᵢ = 1/N)
PPathBump(backend::AbstractADType) =
    PPathBump{Vector{Float64}}(zeros(2), backend)

(path::PPathBump)(t, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        s = t[1] * exp(-(V0 - V1 - t[2])^2)
        return (1 - β) * V0 + β * V1 + (1 - β) * β * s
    end
end

function log_potential(path::PPathBump, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

extract_param(path::PPathBump) = path.t

function extract_reparam(path::PPathBump)
    return path.t
end

function set_param!(path::PPathBump, t::T) where {T <: AbstractVector}
    path.t = t
end
