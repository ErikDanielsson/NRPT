mutable struct SplinePath{T<:AbstractArray, P<:SamplingProblem} <: ParametrizedPath{T, P}
    theta::T
	log_potential::Function
    prep
    backend::AbstractADType
    problem::P
end

get_problem(path::SplinePath) = path.problem

function params_to_knots_spline_path(params::AbstractVector, increasing::Bool)
    summed = [0; cumsum(exp.(params))]
    knots = summed / summed[end]
    return increasing ? knots : 1. .- knots
end

function get_exponents_spline_path(theta::AbstractArray, β) 
    return linear_spline(
        theta_to_eta(theta, [false, true], 2, params_to_knots_spline_path),
        β
    )
end

function SplinePath(n_knots::Int, x0, problem::SamplingProblem, backend::AbstractADType)
    theta0 = ones(2 * n_knots)

    function __log_potential(theta, x, β)
        e1, e2 = get_exponents_spline_path(theta, β)
        return -e1 * V0(problem, x) - e2 * V1(problem, x)
    end

    prep = prepare_path_gradient(__log_potential, theta0, x0, backend)
    return SplinePath(theta0, __log_potential, prep, backend, problem)
end

sample_iid(path::SplinePath) = sample_iid(path.problem)

function log_potential(path::SplinePath, x, β)
    return path.log_potential(path.theta, x, β)
end

function gradient(path::SplinePath, x, β)
    return path_gradient(path.log_potential, path.prep, path.theta, x, β, path.backend)
end

get_exponents(path::SplinePath, β) = get_exponents_spline_path(path.theta, β) 

extract_param(path::SplinePath) = path.theta
extract_reparam(path::SplinePath) = theta_to_eta(path.theta, [false, true])

function set_param!(path::SplinePath, theta::T) where {T <: AbstractArray}
    # TODO: Do the sorting operation here
    path.theta = theta
end