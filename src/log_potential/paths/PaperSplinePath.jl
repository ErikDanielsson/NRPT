mutable struct PaperSplinePath{T<:AbstractArray} <: ParametrizedPath{T}
    theta::T
	log_potential::Function
    prep
    sample_iid::Function
    backend::AbstractADType
end

function params_to_knots(params::AbstractVector, increasing::Bool)
    summed = [0; cumsum(exp.(params))]
    knots = summed / summed[end]
    return increasing ? knots : 1. .- knots
end

function theta_to_eta(theta, increasing::Vector{Bool})
    n_knots = div(length(theta), 2)
    theta_ = reshape(theta, 2, n_knots)
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

function PaperSplinePath(n_knots::Int, x0, problem::SamplingProblem, backend::AbstractADType)
    theta0 = ones(2 * n_knots)

    function __log_potential(theta, x, β)
        eta = theta_to_eta(theta, [false, true])
        l1, l2 = linear_spline(eta, β)
        return -l1 * problem.V0(x) - l2 * problem.V1(x)
    end

    prep = prepare_path_gradient(__log_potential, theta0, x0, backend)
    return PaperSplinePath(theta0, __log_potential, prep, problem.sample_iid, backend)
end

sample_iid(path::PaperSplinePath) = path.sample_iid()

function log_potential(path::PaperSplinePath, x, β)
    return path.log_potential(path.theta, x, β)
end

function gradient(path::PaperSplinePath, x, β)
    return path_gradient(path.log_potential, path.prep, path.theta, x, β, path.backend)
end

get_exponents(path::PaperSplinePath, β) = linear_spline(
    theta_to_eta(path.theta, [false, true]),
    β
)

extract_param(path::PaperSplinePath) = path.theta
extract_reparam(path::PaperSplinePath) = theta_to_eta(path.theta, [false, true])

function set_param!(path::PaperSplinePath, theta::T) where {T <: AbstractArray}
    # TODO: Do the sorting operation here
    path.theta = theta
end