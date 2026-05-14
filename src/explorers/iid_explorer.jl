# For certain classes of models, we might know the distributions
# for each path parameter value explicitly and we are only interested
# in how the index process behaves

abstract type IIDExplorer <: Explorer end

function step(explorer::IIDExplorer, problem::PathProblem, x, β, lp_buff::LP) where {LP <: AbstractVector{Float64}}
    return iid_explore(explorer, problem, β, lp_buff)
end

struct NormalIIDExplorer <: IIDExplorer end

function iid_explore(::NormalIIDExplorer, problem::PathProblem, β, ::LP) where {LP <: AbstractVector{Float64}}
    η0, η1 = get_exponents(problem.path, β)
    μ_β, σ_β = exponents_to_params(problem.problem, η0, η1)
    return rand(Normal(μ_β, σ_β))
end
