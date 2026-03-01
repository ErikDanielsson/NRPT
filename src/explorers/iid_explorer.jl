# For certain classes of models, we might know the distributions
# for each path parameter value explicitly and we are only interested
# in how the index process behaves

abstract type IIDExplorer <: Explorer end

function step(explorer::IIDExplorer, path::Path, x, β) 
    return iid_explore(explorer, path, β)
end

struct GaussianIIDExplorer <: IIDExplorer
    μ0
    σ0
    μ1
    σ1
end


function iid_explore(e::GaussianIIDExplorer, path::Path, β)
    η0, η1 = get_exponents(path, β)
    σ2_β =  (η0 / e.σ0^2 + η1 / e.σ1^2)^(-1)
    μ_β = σ2_β * (η0 * e.μ0 / e.σ0^2 + η1 * e.μ1 / e.σ1^2)
    return rand(Normal(μ_β, sqrt(σ2_β)))   
end