# For certain classes of models, we might know the distributions
# for each path parameter value explicitly and we are only interested
# in how the index process behaves

abstract type IIDExplorer <: Explorer end

function step(explorer::IIDExplorer, path::Path{<:DistributionProblem}, x, β) 
    return iid_explore(explorer, path, β)
end

struct NormalIIDExplorer <: IIDExplorer end

function iid_explore(::NormalIIDExplorer, path::Path{NormalProblem}, β)
    prob = get_problem(path)
    η0, η1 = get_exponents(path, β)
    σ2_β = (η0 / prob.σ0^2 + η1 / prob.σ1^2)^(-1)
    μ_β = σ2_β * (η0 * prob.μ0 / prob.σ0^2 + η1 * prob.μ1 / prob.σ1^2)
    return rand(Normal(μ_β, sqrt(σ2_β)))   
end