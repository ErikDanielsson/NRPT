### Diagonal Adagrad

mutable struct AdagradState{S} <: StochOptState
    eta::Float64
    eps::Float64
    acc_grad::S
end

AdagradState(eta::Float64, eps::Float64) = AdagradState(eta, eps, nothing)
AdagradState{T}(eta::Float64, eps::Float64, ::T) where {T <: Real} =
    AdagradState(eta, eps, 0.0)
AdagradState{T}(eta::Float64, eps::Float64, params0::T) where {T <: AbstractArray} =
    AdagradState(eta, eps, zeros(eltype(T), size(params0)))

function init(problem::PathProblem{<:ParametrizedPath, E}, state::AdagradState{Nothing}) where {E}
    init_acc_grad = extract_param(problem.path)
    return AdagradState(state.eta, state.eps, init_acc_grad)
end

function step!(x, g, state::AdagradState)
    state.acc_grad += g .^ 2
	g_hat = g ./ sqrt.(state.acc_grad .+ state.eps)
	return (state.eta, g_hat)
end

