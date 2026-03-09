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
    σ2_β = (η0 / problem.σ0^2 + η1 / problem.σ1^2)^(-1)
    μ_β = σ2_β * (η0 * problem.μ0 / problem.σ0^2 + η1 * problem.μ1 / problem.σ1^2)
    return rand(Normal(μ_β, sqrt(σ2_β)))   
end