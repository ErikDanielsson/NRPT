function opt_SAA!(
        problem::PathProblem{<:SamplingProblem, P},
        ptchains::PTChains,
        opt_state::NewtonTrustRegionState,
        ::SKLObjective,
        threaded::Bool,
        progress::Bool
    ) where {P <: ParametrizedPath}
    # Construct the loss object, and compute the trust region size
    loss = SNISSKLLoss(problem.path, ptchains, threaded)
    n_samples = size(ptchains)[2]
    rESS_lb = get_lb(opt_state.crit, n_samples)

    return opt_modified_newton_trust_region(
        problem,
        opt_state,
        loss,
        rESS_lb,
        progress
    )
end

