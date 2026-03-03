function theta_to_eta(theta, increasing::Vector{Bool}, components::Int, params_to_knots)
    n_knots = div(length(theta), components)
    theta_ = reshape(theta, components, n_knots)
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