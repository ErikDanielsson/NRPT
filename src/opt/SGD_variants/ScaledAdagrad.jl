### Scaled Adagrad (used in Syed et al. 2021) when parameters are transformed to log-scale

mutable struct ScaledAdagradState{S} <: StochOptState
    eta::Float64
    eps::Float64
    acc_grad::S
    scaler
end

ScaledAdagradState(eta::Float64, eps::Float64, scaler) = ScaledAdagradState(eta, eps, nothing, scaler)
ScaledAdagradState{T}(eta::Float64, eps::Float64, ::T, scaler) where {T <: Real} = ScaledAdagradState(eta, eps, 0.0, scaler)
ScaledAdagradState{T}(eta::Float64, eps::Float64, params0::T, scaler) where {T <: AbstractArray} = ScaledAdagradState(eta, eps, zeros(size(params0)), scaler)

function init(problem::PathProblem{<:ParametrizedPath, E}, state::ScaledAdagradState{Nothing}) where {E}
    init_acc_grad = zeros(size(extract_param(problem.path)))
    return ScaledAdagradState(state.eta, state.eps, init_acc_grad, state.scaler)
end

function step!(x, g, state::ScaledAdagradState)
    # Scale the gradient with its abosolute value
    # and some function of the parameter
    g = g ./ (abs.(g) + state.scaler.(x))
    state.acc_grad += g .^ 2
	g_hat =  g ./ sqrt.(state.acc_grad .+ state.eps)
	return (state.eta, g_hat)
end
