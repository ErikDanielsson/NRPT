mutable struct PaperSplinePath{T<:AbstractArray, P<:SamplingProblem} <: ParametrizedPath{T, P}
    theta::T
	log_potential::Function
    prep
    backend::AbstractADType
    problem::P
end

get_problem(path::PaperSplinePath) = path.problem

function params_to_knots_paper_spline_path(params::AbstractVector, increasing::Bool)
    knots = exp.(params)
    return increasing ? knots : 1. .- knots
end

function get_exponents_paper_spline_path(theta::AbstractArray, β) 
    return linear_spline(
        theta_to_eta(theta, [false, true], 2, params_to_knots_paper_spline_path),
        β
    )
end


function PaperSplinePath(n_knots::Int, x0, problem::SamplingProblem, backend::AbstractADType)
    theta0 = ones(2 * n_knots)

    function __log_potential(theta, x, β)
        e1, e2 = get_exponents_paper_spline_path(theta, β)
        return -e1 * V0(problem, x) - e2 * V1(problem, x)
    end

    prep = prepare_path_gradient(__log_potential, theta0, x0, backend)
    return PaperSplinePath(theta0, __log_potential, prep, backend, problem)
end

sample_iid(path::PaperSplinePath) = sample_iid(path.problem)

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