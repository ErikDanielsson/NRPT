mutable struct PowerPath{T<:Real} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

function PowerPath(t0::T, backend::AbstractADType) where {T <: Real}
    return PowerPath(t0, nothing, backend)
end

(path::PowerPath)(t, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    return (1 - β)^t * V0 + β^t * V1
end

function log_potential(path::PowerPath, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

get_exponents(path::PowerPath, β) = [(1 - β)^path.t, β^path.t]

extract_param(path::PowerPath) = path.t
extract_reparam(path::PowerPath) = path.t
function set_param!(path::PowerPath, t::T) where {T <: Real}
    path.t = t
end
