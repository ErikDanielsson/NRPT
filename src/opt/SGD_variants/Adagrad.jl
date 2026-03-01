### Diagonal AdaGrad

mutable struct AdaGradState{S} <: StochOptState
    eta::Float64
    eps::Float64
    acc_grad::S
end

AdaGradState(eta::Float64, eps::Float64) = AdaGradState(eta, eps, nothing)
AdaGradState(eta::Float64, eps::Float64, ::Float64) = AdaGradState(eta, eps, 0.0)
AdaGradState(eta::Float64, eps::Float64, params0::Vector{Float64}) = AdaGradState(eta, eps, zeros(size(params0)))

function init(problem::PathProblem{<:ParametrizedPath, E}, state::AdaGradState{Nothing}) where {E}
    init_acc_grad = zeros(size(extract_param(problem.path)))
    return AdaGradState(state.eta, state.eps, init_acc_grad)
end

function step!(x, g, state::AdaGradState)
    state.acc_grad += g .^ 2
	g_hat = state.eta * g ./ sqrt.(state.acc_grad .+ state.eps)
	return (η, g_hat)
end

