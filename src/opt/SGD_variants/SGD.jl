###  SGD
struct SGDState <: StochOptState
    eta::Float64
end

function step!(x, g, state::SGDState)
	return (state.eta, g)
end

function init(::PathProblem{ParametrizedPath{P}, E}, state::SGDState) where {P, E}
    return state
end