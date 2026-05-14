# Trust region: take multiple IS-reweighted SKL gradient steps, stopping when
# the ESS ratio (Σwᵢ)²/(n Σwᵢ²) drops below δ on any chain.
function adapt_path!(
        problem::PathProblem{<:SamplingProblem, <:ParametrizedPath},
        ptchains::PTChains{N, T, V},
        schedule,
        opt_state::TrustRegionState,
        ::SKLObjective = SKLObjective(),
    ) where {N, T, V}
    chains = ptchains.chains

    n_chains =

        # Store log_potential(φ₀, lps_i, β) for each chain before any step.
        # IS log weight for sample i in chain n: log w_i = lp(φ, lps_i, β) - ref_lps_n[i]
        ref_lps = Matrix{Float64}(undef, N)
    ref_lps = [log_potential(problem.path, lps)]
    ref_lps = [
        [
                log_potential(problem.path, lps, schedule[chain.index])
                for lps in eachcol(chain.base_potentials)
            ]
            for chain in chains
    ]

    l = sum(
        IS_SKL_loss_chain(problem, chain, schedule, ref)
            for (chain, ref) in zip(chains, ref_lps)
    )

    prog = Progress(opt_state.max_steps; desc = "Trust region optimization", offset = 5)
    for n in 1:opt_state.max_steps
        # Stop if any chain's ESS ratio drops below δ
        for (chain, ref) in zip(chains, ref_lps)
            e = ess_ratio(chain_log_weights(problem, chain, schedule, ref))
            if e < opt_state.δ
                push!(opt_state.n_steps, n)
                return l
            end
        end

        g = sum(
            IS_SKL_grad_chain(problem, chain, schedule, ref)
                for (chain, ref) in zip(chains, ref_lps)
        )

        if nan_grad(g)
            @warn "NaN gradient in trust region update, stopping"
            break
        end

        new_param = step!(extract_param(problem.path), g, opt_state.inner_opt)
        set_param!(problem.path, new_param)

        l = sum(
            IS_SKL_loss_chain(problem, chain, schedule, ref)
                for (chain, ref) in zip(chains, ref_lps)
        )

        next!(
            prog, showvalues = [
                ("objective", l),
                ("η", get_last_eta(opt_state)),
                ("g", g),
            ]
        )
    end

    push!(opt_state.n_steps, opt_state.max_steps)

    return l
end
