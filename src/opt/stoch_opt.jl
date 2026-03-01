abstract type StochOptState end

function init(::PathProblem{<:StaticPath, E}, opt_state::StochOptState) where {E}
    @warn "You are trying to optimize a static path! Ignoring optimizer"
    return opt_state
end




### TODO: Hot DoG like -- DoG with RMSProp accelaration

# mutable struct DoWGState{S} <: StochOptState
#     x0::S
#     acc_grad::Float64
#     reps_rel::Float64
#     max_dist::Union{Float64, Nothing}
#     eps::Float64
# end

# DoWGState(reps_rel, eps) = DoWGState(nothing, 0.0, reps_rel, nothing, eps)

# function init(problem::PathProblem{ParametrizedPath{P}, E}, state::DoWGState{Nothing}) where {P, E}
#     max_dist = state.reps_rel * (1 + sqrt(norm2(problem.path.params)))
#     return DoWGState(problem.path.params, 0.0, state.reps_rel, max_dist, state.eps)
# end

# function step!(x, g, state::DoWGState)
#     state.acc_grad += state.max_dist^2 * norm2(g)
# 	η = state.max_dist^2 / sqrt(state.acc_grad + state.eps)
#     state.max_dist = max(state.max_dist, sqrt(norm2(state.x0 - x)))
# 	return η
# end