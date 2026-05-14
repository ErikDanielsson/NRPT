abstract type Optimizer end

struct NoOptState <: Optimizer end

init(problem, state::NoOptState) = state
step!(x, g, state::NoOptState) = 0.0
get_last_eta(state::NoOptState) = 0.0
get_last_x(state::NoOptState) = nothing

struct ProximalStochOptState{S <: StochOptState, P <: ProximalState, T} <: Optimizer
    stochOptState::S
    proximalState::P
    xs::Vector{T}
    etas::Vector{Float64}
end

get_last_eta(state::ProximalStochOptState) = length(state.etas) > 0 ? state.etas[end] : nothing
get_last_x(state::ProximalStochOptState) = length(state.etas) > 0 ? state.xs[end] : nothing

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

# Trust region optimizer: wraps an inner optimizer and takes multiple IS-reweighted
# gradient steps per round, stopping when the ESS ratio drops below δ.
struct TrustRegionState{O <: ProximalStochOptState} <: Optimizer
    inner_opt::O
    δ::Float64       # minimum ESS ratio to continue stepping: (Σwᵢ)²/(n Σwᵢ²) ≥ δ
    max_steps::Int   # hard cap on inner iterations per round
    n_steps::Vector{Int}
end

TrustRegionState(inner::ProximalStochOptState; δ = 0.5, max_steps = 20) =
    TrustRegionState(inner, Float64(δ), max_steps, Int[])
TrustRegionState(opt::StochOptState, prox::ProximalState; δ = 0.5, max_steps = 20) =
    TrustRegionState(ProximalStochOptState(opt, prox); δ = δ, max_steps = max_steps)
TrustRegionState(opt::StochOptState; δ = 0.5, max_steps = 20) =
    TrustRegionState(ProximalStochOptState(opt); δ = δ, max_steps = max_steps)

get_last_eta(state::TrustRegionState) = get_last_eta(state.inner_opt)

function init(problem::PathProblem{<:SamplingProblem, <:ParametrizedPath}, state::TrustRegionState)
    return TrustRegionState(init(problem, state.inner_opt), state.δ, state.max_steps, Int[])
end
