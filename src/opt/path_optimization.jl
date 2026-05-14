# If we do not have an optimizer, compute the objective loss for display only
function adapt_path!(problem, ptchains::PTChains, ::NoOptState, objective::PathObjective, threaded::Bool, progress::Bool)
    return objective_loss(objective, problem, ptchains, threaded)
end

# If we have a static path do nothing
function adapt_path!(problem::PathProblem{<:SamplingProblem, StaticPath, E}, ptchains::PTChains, schedule, opt_state::ProximalStochOptState, ::PathObjective = SKLObjective()) where {E}
end

function adapt_path!(
        problem::PathProblem{<:SamplingProblem, <:ParametrizedPath, E},
        ptchains::PTChains,
        opt_state::ProximalStochOptState{S, Pr},
        objective::PathObjective,
        threaded::Bool,
        progress::Bool,
    ) where {E, S, Pr}
    l = objective_loss(objective, problem, ptchains, threaded)
    g = objective_gradient(objective, problem, ptchains, threaded)
    if nan_grad(g)
        @warn "Found NaN in gradient, skipping. Loss = $l"
    else
        new_param = step!(extract_param(problem.path), g, opt_state)
        set_param!(problem.path, new_param)
    end
    return l
end
