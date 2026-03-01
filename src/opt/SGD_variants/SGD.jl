###  SGD
struct SGDState <: StochOptState
    eta::Float64
end

function step!(x, g, state::SGDState)
	return (state.eta, g)
end

function init(::PathProblem{<:ParametrizedPath, E}, state::SGDState) where {E}
    return state
end