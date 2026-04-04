mutable struct PaperSplinePath{T<:AbstractArray} <: ParametrizedPath{T}
    theta::T
    prep
    backend::AbstractADType
end

function params_to_knots_paper_spline_path(params::AbstractVector, increasing::Bool)
    if increasing
        knots = [0; exp.(params); 1]
    else
        knots = [1; exp.(params); 0]
    end
    return knots
end

function get_exponents_paper_spline_path(theta::AbstractArray, β)
    return linear_spline(
        theta_to_eta(theta, [false, true], params_to_knots_paper_spline_path),
        β
    )
end

function PaperSplinePath(n_knots::Int, backend::AbstractADType)
    function make_knots(n_knots::Int, increasing::Bool)
        knots = range(0, 1, n_knots + 2)[2:end-1]
        return log.(increasing ? knots : 1 .- knots)
    end
    theta0 = reshape(stack([make_knots(n_knots, false), make_knots(n_knots, true)], dims=1), 2 * n_knots)
    return PaperSplinePath(theta0, nothing, backend)
end

(path::PaperSplinePath)(theta, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    e1, e2 = get_exponents_paper_spline_path(theta, β)
    return e1 * V0 + e2 * V1
end

function log_potential(path::PaperSplinePath, log_potentials::AbstractVector{Float64}, β)
    return path(path.theta, log_potentials, β)
end

get_exponents(path::PaperSplinePath, β) = get_exponents_paper_spline_path(path.theta, β)

extract_param(path::PaperSplinePath) = path.theta
extract_reparam(path::PaperSplinePath) = theta_to_eta(path.theta, [false, true], params_to_knots_paper_spline_path)

function set_param!(path::PaperSplinePath, theta::T) where {T <: AbstractArray}
    # This is the fix monotonicity transformation from the opt. path paper
    n_knots = div(length(theta), 2)
    theta_ = reshape(theta, 2, n_knots)
    theta1 = fix_monotonicity(theta_[1, :], true)
    theta2 = fix_monotonicity(theta_[2, :], true)
    fixed_theta = reshape(stack([theta1, theta2], dims=1), 2 * n_knots)
    path.theta = fixed_theta
end

monosign(v, incr::Bool) = incr ? v : -v
function fix_monotonicity(thetai, increasing::Bool)
    etas = params_to_knots_paper_spline_path(thetai, increasing)
    transform = increasing ? etas : 1 .- etas
    monotone_subset = Int[]
    curval = 0
    for (i, t) in enumerate(transform)
        if curval <= t <= 1.0
            push!(monotone_subset, i)
            curval = t
        end
    end
    # Linearly interpolate between the the valid knots
    new_etas = Float64[]
    if length(monotone_subset) > 1
        for (mi1, mi2) in zip(monotone_subset[1:end-1], monotone_subset[2:end])
            for i in mi1:(mi2-1)
                t = (i - mi1) / (mi2 - mi1)
                x = transform[mi1]
                y = transform[mi2]
                new_eta = x + (y - x) * t
                push!(new_etas, new_eta)
            end
        end
    else
        push!(new_etas, 0.5)
    end
    new_theta = log.(new_etas)[2:end]
    return new_theta
end
