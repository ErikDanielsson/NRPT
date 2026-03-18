# The idea of this path is that we only need to parametrize the exponent of the 
# likelihood since 
mutable struct SingleSplinePath{T<:AbstractArray} <: ParametrizedPath{T}
    theta::T
	log_potential::Function
    prep
    backend::AbstractADType
end

function params_to_knots_single_spline_path(params::AbstractVector, increasing::Bool)
    summed = [0; cumsum(exp.(params))]
    knots = summed / summed[end]
    return increasing ? knots : 1. .- knots
end

function get_exponents_single_spline(theta::AbstractArray, β)
    eta = theta_to_eta(theta, [true], params_to_knots_single_spline_path)
    l2 = linear_spline(eta, β)[1]
    return [1 - β, l2]
end

function SingleSplinePath(n_knots::Int, backend::AbstractADType)
    theta0 = ones(n_knots)

    function __log_potential(theta, log_potentials::Vector{Float64}, β)
        V0, V1 = log_potentials
        e1, e2 = get_exponents_single_spline(theta, β)
        return e1 * V0 + e2 * V1
    end

    prep = prepare_path_gradient(__log_potential, theta0, backend)
    return SingleSplinePath(theta0, __log_potential, prep, backend)
end

function log_potential(path::SingleSplinePath, log_potentials::Vector{Float64}, β)
    return path.log_potential(path.theta, log_potentials, β)
end

function gradient(path::SingleSplinePath, log_potentials::Vector{Float64}, β)
    return path_gradient(path.log_potential, path.prep, path.theta, log_potentials, β, path.backend)
end

get_exponents(path::SingleSplinePath, β) = get_exponents_single_spline(path.theta, β)

extract_param(path::SingleSplinePath) = path.theta
extract_reparam(path::SingleSplinePath) = theta_to_eta(path.theta, [false, true], params_to_knots_single_spline_path)

function set_param!(path::SingleSplinePath, theta::T) where {T <: AbstractArray}
    # TODO: Do the sorting operation here
    path.theta = theta
end