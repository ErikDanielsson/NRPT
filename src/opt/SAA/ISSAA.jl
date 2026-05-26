function opt_SAA!(
        problem::PathProblem{<:SamplingProblem, P},
        ptchains::Ch,
        opt_state::N,
        ::SKLObjective,
        threaded::Bool,
        progress::Bool
    ) where {P <: ParametrizedPath, N <: NewtonTrustRegionState, Ch <: PTChains}
    set_schedule!(problem.path, ptchains.schedule)
    loss = SNISSKLLoss(problem.path, ptchains, threaded)
    
    n_samples = size(ptchains)[2]
    rESS_lb = get_lb(opt_state.crit, n_samples)

    result = opt_modified_newton_trust_region(
        problem,
        opt_state,
        loss,
        rESS_lb,
        progress
    )
    return result
end

