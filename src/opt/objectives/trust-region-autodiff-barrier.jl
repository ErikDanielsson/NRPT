# Trust-region Newton method for the barrier (rejection-rate) loss.
#
# Analogous to trust-region-autodiff.jl (SKL) but replaces the per-chain
# IS-weighted SKL J-function with the Metropolis rejection ratio R.
#
# Key structural difference from the SKL version:
#   SKL  — per-chain IS weights; each chain reweights independently.
#   Barrier — *global* IS weights over the full product distribution of all chains.
#             A single weight per sample index i:
#               log w_i = Σ_n [ path(t, lps_{n,i}, β_n) − ref_lps[n][i] ]
#             This reweights the frozen joint distribution ∏_n p_n(lps_n; t₀)
#             to ∏_n p_n(lps_n; t), which is the natural IS correction for the
#             barrier loss (sum of rejection rates over all adjacent pairs).


# ── R functor ─────────────────────────────────────────────────────────────────

# Metropolis rejection rate for pair (n, n+1), sample pair (lps1, lps2).
# Mirrors R() in rejection-estimator.jl but takes t explicitly for autodiff.
# min(0, ...) is handled transparently by ForwardDiff.
function _R_autodiff(path, t, n::Int, schedule, lps1, lps2)
    prop1 = path(t, lps1, schedule[n + 1])
    prop2 = path(t, lps2, schedule[n])
    ref1  = path(t, lps1, schedule[n])
    ref2  = path(t, lps2, schedule[n + 1])
    return 1 - exp(min(0, prop1 + prop2 - ref1 - ref2))
end


# ── ISBarrierLoss functor ──────────────────────────────────────────────────────

# ref_lps[n][i] = path(t₀, lps_{n,i}, β_n) — frozen before the first inner step.
# Global IS log-weight for sample i: Σ_n [path(t, lps_{n,i}, β_n) − ref_lps[n][i]]
# Loss: L(t) = Σ_{n=1}^{N-1} dot(w̃, [R_n(t, lps_{n,i}, lps_{n+1,i}) for i])
#   where w̃ = softmax(log_w) are the globally normalised IS weights.
struct ISBarrierLoss{P <: ParametrizedPath, C, S, R, T <: Val}
    path::P
    chains::C
    schedule::S
    ref_lps::R
    threaded::T
end

function (loss::ISBarrierLoss{P, C, S, R, Val{false}})(t) where {P <: ParametrizedPath, C, S, R}
    chains   = loss.chains
    schedule = loss.schedule
    N = length(chains)
    M = size(chains[1].base_potentials, 2)

    # Global IS log-weights: sum of per-chain log-potential differences
    log_w = [
        sum(loss.path(t, chains[n].base_potentials[:, i], schedule[n]) - loss.ref_lps[n][i]
            for n in 1:N)
        for i in 1:M
    ]
    w_tilde = softmax(log_w)

    # Sum of IS-weighted rejection rates over all adjacent pairs
    total = zero(eltype(t))
    for n in 1:N-1
        Rvals = [
            _R_autodiff(loss.path, t, n, schedule,
                        chains[n].base_potentials[:, i],
                        chains[n+1].base_potentials[:, i])
            for i in 1:M
        ]
        total += dot(w_tilde, Rvals)
    end
    return total
end

function (loss::ISBarrierLoss{P, C, S, R, Val{true}})(t) where {P <: ParametrizedPath, C, S, R}
    chains   = loss.chains
    schedule = loss.schedule
    N = length(chains)
    M = size(chains[1].base_potentials, 2)

    log_w = [
        sum(loss.path(t, chains[n].base_potentials[:, i], schedule[n]) - loss.ref_lps[n][i]
            for n in 1:N)
        for i in 1:M
    ]
    w_tilde = softmax_(log_w)

    @info (w_tilde)

    function pair_loss(n)
        Rvals = [
            _R_autodiff(loss.path, t, n, schedule,
                        chains[n].base_potentials[:, i],
                        chains[n+1].base_potentials[:, i])
            for i in 1:M
        ]
        return dot(w_tilde, Rvals)
    end
    return tmapreduce(pair_loss, +, 1:N-1; scheduler=:static, init=zero(eltype(t)))
end


# ── ESS helper ────────────────────────────────────────────────────────────────

# Global ESS ratio using the all-chain product IS weights.
# Returns a single number (not a per-chain minimum) since weights are global.
function _min_ess_barrier(path, t, chains, schedule, ref_lps)
    N = length(chains)
    M = size(chains[1].base_potentials, 2)
    log_w = [
        sum(path(t, chains[n].base_potentials[:, i], schedule[n]) - ref_lps[n][i]
            for n in 1:N)
        for i in 1:M
    ]
    return ess_ratio(log_w)
end


# ── Optimizer state ────────────────────────────────────────────────────────────

struct NewtonTrustRegionBarrierState{T, Pr <: ProximalState} <: Optimizer
    prox::Pr
    backend::AbstractADType
    δ::Float64
    max_steps::Int
    λ_reg::Float64
    n_steps::Vector{Int}
    min_eigvals::Vector{Float64}
    xs::T
end

NewtonTrustRegionBarrierState(backend::AbstractADType; δ=0.9, max_steps=20, λ_reg=1e-6) =
    NewtonTrustRegionBarrierState(NoProx(), backend, Float64(δ), max_steps, Float64(λ_reg), Int[], Float64[], [])

NewtonTrustRegionBarrierState(prox::ProximalState, backend::AbstractADType; δ=0.9, max_steps=20, λ_reg=1e-6) =
    NewtonTrustRegionBarrierState(prox, backend, Float64(δ), max_steps, Float64(λ_reg), Int[], Float64[], [])

function init(problem::PathProblem{<:SamplingProblem, <:ParametrizedPath}, state::NewtonTrustRegionBarrierState)
    return NewtonTrustRegionBarrierState(
        state.prox, state.backend, state.δ, state.max_steps, state.λ_reg,
        Int[], Float64[], [extract_param(problem.path)]
    )
end

get_last_eta(s::NewtonTrustRegionBarrierState) = length(s.min_eigvals) > 0 ? s.min_eigvals[end] : nothing


# ── adapt_path! ────────────────────────────────────────────────────────────────

# Newton method with global-ESS backtracking for the barrier loss.
#
# Each inner iteration:
#   1. Compute gradient g and Hessian H of the IS-weighted barrier loss via autodiff.
#   2. Regularise H: λ = max(0, −λ_min(H)) + λ_reg so H + λI is positive definite.
#   3. Newton direction: Δt = −(H + λI)⁻¹ g.
#   4. Backtrack: halve α until global ESS(t + α·Δt) ≥ δ.
#   5. Accept step, update path, recompute g and H, repeat.
#   6. Stop early if no α satisfies the ESS constraint.
function adapt_path!(
    problem::PathProblem{<:SamplingProblem, P},
    ptchains::PTChains,
    schedule,
    opt_state::NewtonTrustRegionBarrierState,
    ::BarrierObjective = BarrierObjective(),
) where {P <: ParametrizedPath}
    τ = 10.0
    chains = ptchains.chains

    # Freeze reference log-potentials at t₀ for all chains before any inner step.
    ref_lps = [
        [log_potential(problem.path, chains[n].base_potentials[:, i], schedule[n])
         for i in 1:size(chains[n].base_potentials, 2)]
        for n in eachindex(chains)
    ]
    n_samples = length(ref_lps[1])

    loss = ISBarrierLoss(problem.path, chains, schedule, ref_lps, Val(false))

    t = extract_param(problem.path)
    l = loss(t)

    prog = Progress(opt_state.max_steps; desc="Newton trust region barrier (autodiff)", offset=5, enabled=true)
    ProgressMeter.update!(prog, 0, force=true, showvalues=[
        ("objective", l),
    ])

    g = DifferentiationInterface.gradient(loss, opt_state.backend, t) + τ * t / n_samples
    H = DifferentiationInterface.hessian(loss, opt_state.backend, t) + τ * I / n_samples

    for n in 1:opt_state.max_steps
        if nan_grad(g) || any(isnan, H)
            @warn "NaN in gradient or Hessian during Newton barrier trust region update, stopping"
            @info l
            @info g
            @info H
            break
        end

        min_eig = minimum(eigvals(Symmetric(H)))
        push!(opt_state.min_eigvals, min_eig)

        Δt = _newton_step(g, H, min_eig, opt_state.λ_reg)
        if sqrt(norm2(Δt)) < 1e-16
            ProgressMeter.update!(prog, force=true, showvalues=[
                ("objective", l),
                ("Δt", Δt),
            ])
            return l
        end

        # Backtracking: halve α until global ESS ≥ δ.
        α     = 1.0
        new_t = nothing
        min_e = 0.0
        while α > 2^-20
            candidate = step!(t + α * Δt, opt_state.prox)
            min_e = _min_ess_barrier(problem.path, candidate, chains, schedule, ref_lps)
            if min_e ≥ opt_state.δ
                new_t = candidate
                break
            end
            α /= 2
        end

        if new_t === nothing
            # println("No new t")
            push!(opt_state.n_steps, n)
            return l
        end
        # println("Setting param: $g, $H")
        set_param!(problem.path, new_t)
        push!(opt_state.xs, copy(new_t))
        l = loss(new_t)

        next!(prog, showvalues=[
            ("objective", l),
            ("rESS",      min_e),
            ("α",         α),
            ("||g||",     sqrt(norm2(g))),
            ("λ_min(H)",  min_eig),
        ])

        t = extract_param(problem.path)
        g = DifferentiationInterface.gradient!(loss, g, opt_state.backend, t) + τ * t / n_samples
        H = DifferentiationInterface.hessian!(loss, H, opt_state.backend, t) + τ * I / n_samples
        
    end

    push!(opt_state.n_steps, opt_state.max_steps)
    return l
end
