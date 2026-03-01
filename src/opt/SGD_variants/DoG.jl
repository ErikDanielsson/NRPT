### DoG -- Distance over Gradients
mutable struct DoGState{S} <: StochOptState
    x0::S
    acc_grad::Float64
    reps_rel::Float64
    max_dist::Union{Float64, Nothing}
    eps::Float64
end

DoGState(reps_rel, eps) = DoGState(nothing, 0.0, reps_rel, nothing, eps)

function init(problem::PathProblem{<:ParametrizedPath, E}, state::DoGState{Nothing}) where {E}
    param0 = extract_param(problem.path)
    max_dist = state.reps_rel * (1 + sqrt(norm2(param0)))
    return DoGState(param0, 0.0, state.reps_rel, max_dist, state.eps)
end

function step!(x, g, state::DoGState)
    state.acc_grad += norm2(g)
	η = state.max_dist / sqrt(state.acc_grad + state.eps)
    state.max_dist = max(state.max_dist, sqrt(norm2(state.x0 - x)))
	return (η, g)
end
