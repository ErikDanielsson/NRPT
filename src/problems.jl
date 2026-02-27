struct SamplingProblem{T, S, W, L, D}
    V0::T
    sample_iid::S
    V::W
    data::Vector{D}
    V1::L
end

function SamplingProblem(V0, sample_iid, V, data)
    function V1(x)
        sum(V(x, d) for d in data)
    end
    return SamplingProblem(
        V0, sample_iid, V, data, V1
    )
end

abstract type Path end

struct StaticPath <: Path
	log_potential
    sample_iid
end

sample_iid(path::StaticPath) = path.sample_iid()

function log_potential(path::StaticPath, x, β::T) where {T <: Real}
    return path.log_potential(x, β)
end


function linear_path(problem::SamplingProblem)
    function log_potential(x, β)
        return -((1 - β) * problem.V0(x) + β * problem.V1(x))
    end
    return StaticPath(log_potential, problem.sample_iid) 
end

mutable struct ParametrizedPath{T} <: Path
    params::T
	log_potential::Function
    prep
    sample_iid::Function
    backend::AbstractADType
end

function ParametrizedPath(params::Float64, xlike::T, log_potential::Function, sample_iid::Function, backend::AbstractADType) where {T}
    @info "Preparing path derivative"
    prep = DifferentiationInterface.prepare_derivative(
        log_potential,
        backend,
        params,
        DifferentiationInterface.Constant(xlike),
        DifferentiationInterface.Constant(1.0)
    )
    return ParametrizedPath(params, log_potential, prep, sample_iid, backend)
end

function gradient(path::ParametrizedPath{Float64}, params, x, β)
    return DifferentiationInterface.derivative(
        path.log_potential,
        path.prep,
        path.backend,
        params,
        DifferentiationInterface.Constant(x),
        DifferentiationInterface.Constant(β)
    )
end

function ParametrizedPath(params, xlike::T, log_potential::Function, sample_iid::Function, backend::AbstractADType) where {T}
    @info "Preparing path gradient"
    prep = DifferentiationInterface.prepare_gradient(
        log_potential,
        backend,
        similar(params),
        DifferentiationInterface.Constant(xlike),
        DifferentiationInterface.Constant(1.0)
    )
    return ParametrizedPath(params, log_potential, prep, sample_iid, backend)
end


function gradient(path::ParametrizedPath{<:AbstractArray}, params, x, β)
    return DifferentiationInterface.gradient(
        path.log_potential,
        path.prep,
        path.backend,
        params,
        DifferentiationInterface.Constant(x),
        DifferentiationInterface.Constant(β)
    )
end

sample_iid(path::ParametrizedPath) = path.sample_iid()

function log_potential(path::ParametrizedPath, x, β::T) where {T <: Real}
    return path.log_potential(path.params, x, β)
end

function power_path(params0, x0, problem::SamplingProblem, backend::AbstractADType)
    function __log_potential(params, x, β)
        return -((1 - β)^params * problem.V0(x) + β^params * problem.V1(x))
    end
    function gradient_(params, x, β)
        if β <= 0.0 || β >= 1.0
            return 0.0
        end
        return -(log(1 - β) * (1 - β)^params * problem.V0(x) + log(β) * β^params * problem.V1(x))
    end

    return ParametrizedPath(params0, x0, __log_potential,  problem.sample_iid, backend) 
end

# These are just to keep track of things if we come up with some better parametrization...
function q_to_param(q)
    return logit(q)
end

function param_to_q(param)
    return logistic(param)
end

function q_path(q0, x0, problem::SamplingProblem, backend::AbstractADType)
    params0 = q_to_param(q0)
    function __log_potential(param, x, β)
        q = param_to_q(param) 
        p = 1 - q
        return logweightaddexp(1 - β, -p * problem.V0(x), β, -p * problem.V1(x)) / p
    end

    return ParametrizedPath(params0, x0, __log_potential,  problem.sample_iid, backend) 
end

function params_to_knots(params::AbstractVector, increasing::Bool)
    # p = [exp.(params); 1.]
    summed = [0; cumsum(exp.(params))]
    knots = summed / summed[end]
    return increasing ? knots : 1. .- knots
end

function theta_to_eta(theta, increasing::Vector{Bool})
    theta_ = reshape(theta, 2, div(length(theta), 2))
    eta = stack(map(((r, i),) -> params_to_knots(r, i), zip(eachrow(theta_), increasing)), dims=1)
    return eta
end

function linear_spline(eta, β::Float64)
    if β == 0.0
        return eta[:, 1]
    else
        K = size(eta, 2) - 1
        k = ceil(Int, K * β)
        return eta[:, k] * (k - K * β) + eta[:, k+1] * (K * β - k + 1)
    end
end

function spline_path(n_knots, x0, problem::SamplingProblem, backend::AbstractADType)
    theta0 = ones(2 * n_knots)

    function __log_potential(theta, x, β)
        eta = theta_to_eta(theta, [false, true])
        l1, l2 = linear_spline(eta, β)
        return -l1 * problem.V0(x) - l2 * problem.V1(x)
    end

    return ParametrizedPath(theta0, x0, __log_potential,  problem.sample_iid, backend) 
end