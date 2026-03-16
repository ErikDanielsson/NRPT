# If we do not have an optimizer do nothing
function adapt_path!(problem, ptchains::PTChains, schedule, opt_state::NoOptState)
    return sum(mcmc_loss(problem, chain, schedule) for chain in ptchains.chains)
end

# If we have a static path do nothing
function adapt_path!(problem::PathProblem{<:SamplingProblem, StaticPath, E}, ptchains::PTChains, schedule, opt_state::ProximalStochOptState) where {E}
end

function adapt_path!(
    problem::PathProblem{<:SamplingProblem, <:ParametrizedPath, E},
    ptchains::PTChains,
    schedule,
    opt_state::ProximalStochOptState{S, Pr}
) where {E, S, Pr}
    l = SKL_loss(problem, ptchains, schedule)
    g = SKL_gradient(problem, ptchains, schedule)
    new_param = step!(extract_param(problem.path), g, opt_state)
    set_param!(problem.path, new_param)
    return l
end