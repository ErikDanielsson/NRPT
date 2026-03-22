
abstract type LLElem end

struct LinearLLElem <: LLElem end

function log_potential(::LinearLLElem, V1, β)
    return β * V1
end

struct PowerMeanLLElem <: LLElem
    p
end

function log_potential(elem::PowerMeanLLElem, V1, β)
    if V1 == -Inf
        return log(1 - β) / elem.p
    else
        return logweightmeanexp(1 - β, 1.0, β, elem.p * V1)
    end
end

mutable struct GPath{T<:AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    h::Vector{LLElem} 
	log_potential::Function
    prep
    backend::AbstractADType
end

function ω_to_t(ω)
    return log.(ω[2:end]) .- log(ω[1])
end

function t_to_ω(t)
    t_ = [0.0; t]
    softmax(t_)
end

function GPath(ω0::T, backend::AbstractADType) where {T <: AbstractVector{<:Real}}
    t0 = ω_to_t(ω0)
    function __log_potential(t, log_potentials::AbstractVector{Float64}, β)
        V0, V1 = log_potentials
        q = param_to_q(t) 
        p = 1 - q
        if β == 0.0
            return V0 
        elseif β == 1.0
            return V1
        else
            return V0 + logweightaddexp(1 - β, 1.0, β, p * (V1 - V0)) / p
        end
    end
    prep = prepare_path_gradient(__log_potential, t0, backend)
    return GPath(t0, __log_potential, prep, backend) 
end

function gradient(path::GPath, log_potentials::AbstractVector{Float64}, β)
    return path_gradient(path.log_potential, path.prep, path.t, log_potentials, β, path.backend)
end

function log_potential(path::GPath, log_potentials::AbstractVector{Float64}, β)
    return path.log_potential(path.t, log_potentials, β)
end

extract_param(path::GPath) = path.t
extract_reparam(path::GPath) = param_to_q(path.t)

function set_param!(path::GPath, t::T) where {T <: Real}
    path.t = t
end