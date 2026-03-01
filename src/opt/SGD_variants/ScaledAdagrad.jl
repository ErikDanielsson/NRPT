### Scaled AdaGrad (used in Syed et al. 2021) when parameters are transformed to log-scale

mutable struct ScaledAdaGradState{S} <: StochOptState
    eta::Float64
    eps::Float64
    acc_grad::S
    param_scaler
end

ScaledAdaGradState(eta::Float64, eps::Float64, scaler) = ScaledAdaGradState(eta, eps, nothing, scaler)
ScaledAdaGradState(eta::Float64, eps::Float64, ::Float64, scaler) = ScaledAdaGradState(eta, eps, 0.0, scaler)
ScaledAdaGradState(eta::Float64, eps::Float64, params0::Vector{Float64}, scaler) = ScaledAdaGradState(eta, eps, zeros(size(params0)), scaler)

function init(problem::PathProblem{<:ParametrizedPath, E}, state::ScaledAdaGradState{Nothing}) where {E}
    init_acc_grad = zeros(size(extract_param(problem.path)))
    return ScaledAdaGradState(state.eta, state.eps, init_acc_grad, state.scaler)
end

function step!(x, g, state::ScaledAdaGradState)
    # Scale the gradient with its abosolute value
    # and some function of the parameter
    g = g / (abs.(g) + state.scaler.(x))
    state.acc_grad += g .^ 2
	g_hat = state.eta * g ./ sqrt.(state.acc_grad .+ state.eps)
	return (η, g_hat)
end

