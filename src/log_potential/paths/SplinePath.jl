mutable struct SplinePath{T<:AbstractArray} <: ParametrizedPath{T}
    theta::T
    prep
    backend::AbstractADType
end

function params_to_knots_spline_path(params::AbstractVector, increasing::Bool)
    rel_params = [params; 1.0]
    summed = [0; cumsum(exp.(rel_params))]
    knots = summed / summed[end]
    return increasing ? knots : 1. .- knots
end

function get_exponents_spline_path(theta::AbstractArray, β)
    return linear_spline(
        theta_to_eta(theta, [false, true], params_to_knots_spline_path),
        β
    )
end

function SplinePath(n_knots::Int, backend::AbstractADType)
    return SplinePath(ones(2 * n_knots), nothing, backend)
end

(path::SplinePath)(theta, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    e1, e2 = get_exponents_spline_path(theta, β)
    return e1 * V0 + e2 * V1
end

function log_potential(path::SplinePath, log_potentials::AbstractVector{Float64}, β)
    return path(path.theta, log_potentials, β)
end

get_exponents(path::SplinePath, β) = get_exponents_spline_path(path.theta, β)

extract_param(path::SplinePath) = path.theta
extract_reparam(path::SplinePath) = theta_to_eta(path.theta, [false, true], param_to_knots_spline_path)

function set_param!(path::SplinePath, theta::T) where {T <: AbstractArray}
    path.theta = theta
end
