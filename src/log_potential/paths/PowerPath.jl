mutable struct PowerPath{T<:Real} <: ParametrizedPath{T}
    t::T
	log_potential::Function
    prep
    backend::AbstractADType
end

function PowerPath(t0::T, backend::AbstractADType) where {T <: Real}
    function __log_potential(t, log_potentials::AbstractVector{Float64}, β)
        V0, V1 = log_potentials
        return -((1 - β)^t * V0 + β^t * V1)
    end
    prep = prepare_path_gradient(__log_potential, t0, backend)
    return PowerPath(t0, __log_potential, prep, backend)
end

function log_potential(path::PowerPath, log_potentials::AbstractVector{Float64}, β)
    return path.log_potential(path.t, log_potentials, β)
end

function gradient(path::PowerPath, log_potentials::AbstractVector{Float64}, β)
    return path_gradient(path.log_potential, path.prep, path.t, log_potentials, β, path.backend)
end

get_exponents(path::PowerPath, β) = ((1 - β)^path.t, β^path.t)

extract_param(path::PowerPath) = path.t
extract_reparam(path::PowerPath) = path.t
function set_param!(path::PowerPath, t::T) where {T <: Real}
    path.t = t
end