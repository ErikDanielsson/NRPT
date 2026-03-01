struct QPath{T<:Real} <: ParametrizedPath{T}
    t::T
	log_potential::Function
    prep
    sample_iid::Function
    backend::AbstractADType
end

# These are just to keep track of things if we come up with some better parametrization...
function q_to_param(q)
    return logit(q)
end

function param_to_q(param)
    return logistic(param)
end

function QPath(q0::T, x0, problem::SamplingProblem, backend::AbstractADType) where {T <: Real}
    t0 = q_to_param(q0)
    function __log_potential(t, x, β)
        q = param_to_q(param) 
        p = 1 - q
        return logweightaddexp(1 - β, -p * problem.V0(x), β, -p * problem.V1(x)) / p
    end
    prep = prepare_path_gradient(__log_potential, t0, x0, backend)
    return QPath(t0, __log_potential, prep, problem.sample_iid, backend) 
end

function gradient(path::QPath, x, β)
    return path_gradient(path.log_potential, path.prep, path.t, x, β, path.backend)
end

sample_iid(path::QPath) = path.sample_iid()

function log_potential(path::QPath, x, β)
    return path.log_potential(path.t, x, β)
end

extract_param(path::QPath) = path.t
extract_reparam(path::QPath) = param_to_q(path.t)

function set_param!(path::QPath, t::T) where {T <: Real}
    path.t = t
end