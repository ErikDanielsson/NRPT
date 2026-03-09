###  SGD
struct SGDState <: StochOptState
    eta::Float64
end

function step!(x, g, state::SGDState)
	return (state.eta, g)
end

function init(::PathProblem{P, <:ParametrizedPath, E}, state::SGDState) where {P, E}
    return state
end