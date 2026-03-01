### ADAM

struct AdamState{T} <: StochOptState
    eta::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    t::Float64
    m::T
    v::T
end

AdamState(eta, beta1, beta2, eps) = AdamState(eta, beta1, beta2, eps, 0.0, nothing, nothing)
AdamState(eta, beta1, beta2, eps, ::Float64) = AdamState(eta, beta1, beta2, eps, 0.0, 0.0, 0.0)
AdamState(eta, beta1, beta2, eps, params0::Vector{Float64}) = AdamState(
    eta, beta1, beta2, eps, 0.0, zeros(size(params0)), zeros(size(params0))
)

function step!(x, g, state::AdamState)
    # Compute new state
    m_t = state.beta1 * state.m + (1 - state.beta1) * g
    v_t = state.beta2 * state.v + (1 - state.beta2) * g .^ 2
    m_hat_t = m_t / (1 - state.beta1^state.t)
    v_hat_t = v_t / (1 - state.beta2^state.t)
    g_hat = m_hat_t / (sqrt.(v_hat_t) .+ state.eps)

    # Update state
    state.m = m_t
    state.v = v_t
    state.t += 1

	return (state.eta, g_hat)
end

function init(::PathProblem{<:ParametrizedPath, E}, state::AdamState{Nothing}) where {E}
    return state
end
