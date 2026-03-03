mutable struct PowerPath{T<:Real,P<:SamplingProblem} <: ParametrizedPath{T, P}
    t::T
	log_potential::Function
    prep
    backend::AbstractADType
    problem::P
end

get_problem(path::PowerPath) = path.problem

function PowerPath(t0::T, x0, problem::SamplingProblem, backend::AbstractADType) where {T <: Real}
    function __log_potential(t, x, β)
        return -((1 - β)^t * V0(problem, x) + β^t * V1(problem, x))
    end
    prep = prepare_path_gradient(__log_potential, t0, x0, backend)
    return PowerPath(t0, __log_potential, prep, backend, problem)
end

sample_iid(path::PowerPath) = sample_iid(path.problem)

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