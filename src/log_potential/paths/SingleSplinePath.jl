# The idea of this path is that we only need to parametrize the exponent of the 
# likelihood since 
mutable struct SingleSplinePath{T<:AbstractArray, P<:SamplingProblem} <: ParametrizedPath{T, P}
    theta::T
	log_potential::Function
    prep
    backend::AbstractADType
    problem::P
end

get_problem(path::SingleSplinePath) = path.problem

function params_to_knots_single_spline_path(params::AbstractVector, increasing::Bool)
    summed = [0; cumsum(exp.(params))]
    knots = summed / summed[end]
    return increasing ? knots : 1. .- knots
end

function get_exponents_single_spline(theta::AbstractArray, β)
    eta = theta_to_eta(theta, [true], 1, params_to_knots_single_spline_path)
    l2 = linear_spline(eta, β)[1]
    return (1 - β, l2)
end

function SingleSplinePath(n_knots::Int, x0, problem::SamplingProblem, backend::AbstractADType)
    theta0 = ones(n_knots)

    function __log_potential(theta, x, β)
        e1, e2 = get_exponents_single_spline(theta, β)
        return -e1 * V0(problem, x) - e2 * V1(problem, x)
    end

    prep = prepare_path_gradient(__log_potential, theta0, x0, backend)
    return SingleSplinePath(theta0, __log_potential, prep, backend, problem)
end

sample_iid(path::SingleSplinePath) = sample_iid(path.problem)

function log_potential(path::SingleSplinePath, x, β)
    return path.log_potential(path.theta, x, β)
end

function gradient(path::SingleSplinePath, x, β)
    return path_gradient(path.log_potential, path.prep, path.theta, x, β, path.backend)
end

get_exponents(path::SingleSplinePath, β) = get_exponents_single_spline(path.theta, β)

extract_param(path::SingleSplinePath) = path.theta
extract_reparam(path::SingleSplinePath) = theta_to_eta(path.theta, [false, true])

function set_param!(path::SingleSplinePath, theta::T) where {T <: AbstractArray}
    # TODO: Do the sorting operation here
    path.theta = theta
end