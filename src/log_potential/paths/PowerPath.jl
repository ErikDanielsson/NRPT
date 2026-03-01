struct PowerPath{T<:Real} <: ParametrizedPath{T}
    t::T
	log_potential::Function
    prep
    sample_iid::Function
    backend::AbstractADType
end

function PowerPath(t0::T, x0, problem::SamplingProblem, backend::AbstractADType) where {T <: Real}
    function __log_potential(t, x, β)
        return -((1 - β)^t * problem.V0(x) + β^t * problem.V1(x))
    end
    prep = prepare_path_gradient(__log_potential, t0, x0, backend)
    return PowerPath(t0, __log_potential, prep, problem.sample_iid, backend)
end

sample_iid(::PowerPath) = path.sample_iid()

function log_potential(path::PowerPath, x, β)
    return path.log_potential(path.t, x, β)
end

function gradient(path::PowerPath, x, β)
    return path_gradient(path.log_potential, path.prep, path.t, x, β, path.backend)
end

get_exponents(path::PowerPath, β) = ((1 - β)^path.t, β^path.t)

extract_param(path::PowerPath) = path.t
extract_reparam(path::PowerPath) = path.t
function set_param!(path::PowerPath, t::T) where {T <: Real}
    path.t = t
end