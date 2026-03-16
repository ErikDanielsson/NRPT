# For certain classes of models, we might know the distributions
# for each path parameter value explicitly and we are only interested
# in how the index process behaves

abstract type IIDExplorer <: Explorer end

function step(explorer::IIDExplorer, problem::PathProblem, x, β) 
    return iid_explore(explorer, problem.path, problem.problem, β)
end

struct NormalIIDExplorer <: IIDExplorer end

function iid_explore(::NormalIIDExplorer, path::Path, problem::NormalProblem, β)
    η0, η1 = get_exponents(path, β)
    μ_β, σ_β = exponents_to_params(problem, η0, η1)
    return rand(Normal(μ_β, σ_β))   
end