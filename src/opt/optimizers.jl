abstract type Optimizer end

struct NoOptState <: Optimizer end

init(problem, state::NoOptState) = state 
step!(x, g, state::NoOptState) = 0.0
get_last_eta(state::NoOptState) = 0.0
get_last_x(state::NoOptState) = nothing 

struct ProximalStochOptState{S <: StochOptState, P <: ProximalState, T}  <: Optimizer
    stochOptState::S
    proximalState::P
    xs::Vector{T}
    etas::Vector{Float64}
end

get_last_eta(state::ProximalStochOptState) = state.etas[end]
get_last_x(state::ProximalStochOptState) = state.xs[end]

ProximalStochOptState(s, p) = ProximalStochOptState(s, p, [], Float64[])
ProximalStochOptState(s) = ProximalStochOptState(s, NoProx())

function init(problem::PathProblem{P, <:ParametrizedPath, E}, state::ProximalStochOptState) where {P, E}
    return ProximalStochOptState(init(problem, state.stochOptState), state.proximalState, [extract_param(problem.path)], Float64[])
end

function step!(x, g, state::ProximalStochOptState{S, P}) where {S, P}
    # Take a gradient step
    η, g_hat = step!(x, g, state.stochOptState)
    x = x - η * g_hat
    # Take a proximal step, this is typically a projection onto a feasible region
    x = step!(x, state.proximalState)
    push!(state.xs, x)
    push!(state.etas, η)
	return x
end