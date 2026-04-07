# Trust-region Newton method using DifferentiationInterface for autodiff.
#
# Rather than using the analytically derived gradient formulas (∇J, ∇W
# from trust_region.jl / SKL.jl), we differentiate the total IS-weighted
# SKL loss L(t) directly with respect to the path parameter t.  Autodiff
# handles the IS-weight correction automatically, so no manual Cov(∇W, J)
# term is needed.  The Hessian is computed in the same pass, enabling a
# regularised Newton step in place of a first-order SGD update.
#
# Each ParametrizedPath already implements (path::P)(t, lps, β) -> scalar,
# so the path itself is the callable used for autodiff — no wrapper needed.


# ── Building the autodiffable loss ────────────────────────────────────────

# J_n(t, e_i): local SKL contribution for chain n, sample i.
# Mirrors the J function in SKL.jl but takes t explicitly for autodiff.
function _J_autodiff(path, t, n::Int, schedule, lps)
    N = length(schedule)
    if n == 1
        return path(t, lps, schedule[1]) - path(t, lps, schedule[2])
    elseif n == N
        return path(t, lps, schedule[N]) - path(t, lps, schedule[N-1])
    else
        return (2path(t, lps, schedule[n])
                - path(t, lps, schedule[n+1])
                - path(t, lps, schedule[n-1]))
    end
end

# Functor holding the IS-weighted SKL loss.
# ref_lps[k][i] = log_potential(path, lps_i, β_k) frozen at φ₀ before the
# first inner step; the IS weight for sample i in chain k is
#   log wᵢ = path(t, lps_i, βₖ) − ref_lps[k][i].
struct ISSKLLoss{P <: ParametrizedPath, C, S, R}
    path::P
    chains::C
    schedule::S
    ref_lps::R
end

function (loss::ISSKLLoss)(t)
    total = zero(eltype(t))
    function chain_loss(args)
        chain, ref = args
        n = chain.index
        β = loss.schedule[n]
        w = softmax([loss.path(t, lps, β) - ref[i]
                 for (i, lps) in enumerate(eachcol(chain.base_potentials))])
        Js = [_J_autodiff(loss.path, t, n, loss.schedule, lps)
              for lps in eachcol(chain.base_potentials)]
        return dot(w, Js)
    end
    # for args in zip(loss.chains, loss.ref_lps)
    #     total += chain_loss(args)
    # end
    total = tmapreduce(chain_loss, +, collect(zip(loss.chains, loss.ref_lps)); scheduler=:static, init=0.0)
    return total
end


# ── Newton step with Hessian regularisation ────────────────────────────────

# Regularised Newton step:  Δt = −(H + λI)⁻¹ g
#
# λ is chosen as  max(0, −λ_min(H)) + λ_reg  so that H + λI is positive
# definite with the smallest possible shift.  λ_reg > 0 is a fixed floor
# that also handles the (rare) case where H is exactly singular.
function _newton_step(g::Vector{Float64}, H::Matrix{Float64}, min_eig::Float64, λ_reg::Float64)
    λ = max(0.0, -min_eig) + λ_reg
    return (Symmetric(H) + λ * I) \ (-g)
end


# ── Optimizer state ────────────────────────────────────────────────────────

struct NewtonTrustRegionState{T, Pr <: ProximalState} <: Optimizer
    prox::Pr
    backend::AbstractADType
    δ::Float64          # ESS ratio threshold: stop inner loop when any chain drops below this
    max_steps::Int      # hard cap on inner Newton iterations per round
    λ_reg::Float64      # Tikhonov regularisation floor added to the Hessian diagonal
    n_steps::Vector{Int}
    min_eigvals::Vector{Float64}  # minimum eigenvalue of H each step (negative = non-convex)
    xs::T 
end

NewtonTrustRegionState(backend::AbstractADType; δ=0.9, max_steps=20, λ_reg=1e-6) =
    NewtonTrustRegionState(NoProx(), backend, Float64(δ), max_steps, Float64(λ_reg), Int[], Float64[], [])

NewtonTrustRegionState(prox::ProximalState, backend::AbstractADType; δ=0.9, max_steps=20, λ_reg=1e-6) =
    NewtonTrustRegionState(prox, backend, Float64(δ), max_steps, Float64(λ_reg), Int[], Float64[], [])

function init(problem::PathProblem{<:SamplingProblem, <:ParametrizedPath}, state::NewtonTrustRegionState)
    return NewtonTrustRegionState(state.prox, state.backend, state.δ, state.max_steps, state.λ_reg, Int[], Float64[], [extract_param(problem.path)])
end

get_last_eta(ntrs::NewtonTrustRegionState) = length(ntrs.min_eigvals) > 0 ? ntrs.min_eigvals[end] : nothing


# ── ESS helper ────────────────────────────────────────────────────────────

# Minimum ESS ratio across all chains at a candidate parameter value t.
# Uses the path functor directly — no set_param! required.
function _min_ess(path, t, chains, schedule, ref_lps)
    return minimum(
        ess_ratio([path(t, lps, schedule[chain.index]) - ref[i]
                   for (i, lps) in enumerate(eachcol(chain.base_potentials))])
        for (chain, ref) in zip(chains, ref_lps)
    )
end


# ── adapt_path! ────────────────────────────────────────────────────────────

# Newton method with ESS-based backtracking line search.
#
# Each inner iteration:
#   1. Compute gradient g and Hessian H of the IS-weighted SKL loss.
#   2. Compute the full Newton direction Δt = −(H + λI)⁻¹ g.
#   3. Backtrack: try α = 1, 1/2, 1/4, … until ESS(t + α·Δt) ≥ δ.
#      The path functor evaluates candidates without mutating path.t,
#      so set_param! is only called once the step is accepted.
#   4. If no α satisfies the ESS constraint, stop.
function adapt_path!(
    problem::PathProblem{<:SamplingProblem, P},
    ptchains::PTChains,
    schedule,
    opt_state::NewtonTrustRegionState,
    ::SKLObjective = SKLObjective(),
) where {P <: ParametrizedPath}
    chains = ptchains.chains

    # Freeze reference log-potentials at the current φ₀ before any inner step.
    ref_lps = [
        [log_potential(problem.path, lps, schedule[chain.index])
         for lps in eachcol(chain.base_potentials)]
        for chain in chains
    ]

    loss = ISSKLLoss(problem.path, chains, schedule, ref_lps)

    t = extract_param(problem.path)
    l = loss(t)

    prog = Progress(opt_state.max_steps; desc="Newton trust region (autodiff)", offset=5)
    ProgressMeter.update!(prog, 0, force=true, showvalues=[
        ("objective", l),
    ])
    t = extract_param(problem.path)
    g = DifferentiationInterface.gradient(loss, opt_state.backend, t)
    H = DifferentiationInterface.hessian(loss, opt_state.backend, t)
    for n in 1:opt_state.max_steps
       if sqrt(norm2(g)) < 1e-16
            return l
        end
        if nan_grad(g) || any(isnan, H)
            @warn "NaN in gradient or Hessian during Newton trust region update, stopping"
            break
        end

        min_eig = minimum(eigvals(Symmetric(H)))
        push!(opt_state.min_eigvals, min_eig)

        Δt = _newton_step(g, H, min_eig, opt_state.λ_reg)

        # Backtracking line search: halve α until ESS ≥ δ or α is negligible.
        α     = 1.0
        new_t = nothing
        min_e = 0.0
        while α > 2.0^-10
            candidate = step!(t + α * Δt, opt_state.prox)
            min_e = _min_ess(problem.path, candidate, chains, schedule, ref_lps)
            if min_e ≥ opt_state.δ
                new_t = candidate
                break
            end
            α /= 2
        end

        if new_t === nothing
            push!(opt_state.n_steps, n)
            return l
        end

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
        g = DifferentiationInterface.gradient!(loss, g, opt_state.backend, t)
        H = DifferentiationInterface.hessian!(loss, H, opt_state.backend, t)
     
    end

    push!(opt_state.n_steps, opt_state.max_steps)
    return l
end
