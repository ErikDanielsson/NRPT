# Importance sampling - sample average approximation - loss
# Uses forward autodiff through SNIS to compute gradients and Hessians
# at 


function J_fast(t, path, w, target_lps, ptchains::PTChains{N, V}, i::Int) where {N, V}
    total = zero(eltype(t))
    if i == 1
        @views @inbounds for j in eachindex(w, target_lps)
            base_lps = base_potentials(ptchains, i, j)
            total += w[j] * (target_lps[j] - path(t, base_lps, ptchains.schedule[i + 1]))
        end
        return total
    elseif i == N
        @views @inbounds for j in eachindex(w, target_lps)
            base_lps = base_potentials(ptchains, i, j)
            total += w[j] * (target_lps[j] - path(t, base_lps, ptchains.schedule[i - 1]))
        end
        return total
    else
        @views @inbounds for j in eachindex(w, target_lps)
            base_lps = base_potentials(ptchains, i, j)
            lp_pos = path(t, base_lps, ptchains.schedule[i + 1])
            lp_neg = path(t, base_lps, ptchains.schedule[i - 1])
            total += w[j] * (2target_lps[j] - lp_pos - lp_neg)
        end
        return total
    end
end

struct ISSKLLoss{P <: ParametrizedPath, Tr <: Val, PT <: PTChains}
    path::P
    ptchains::PT
    ref_lps::Matrix{Float64}
end

function ISSKLLoss(path::P, ptchains::PT, threaded=false) where {P <: ParametrizedPath, PT <: PTChains}
    n_chains, iterations = size(ptchains)
    # Compute the log potential at ϕ_0
    ref_lps = Matrix{Float64}(undef, iterations, n_chains)
    for i in 1:n_chains
        β = ptchains.schedule[i]
        for j in 1:iterations
            ref_lps[j, i] = log_potential(path, base_potentials(ptchains, i, j), β)
        end
    end
    return ISSKLLoss{P, Val{threaded}, PT}(path, ptchains, ref_lps)
end

function (loss::ISSKLLoss{P, Val{false}, PT})(t::S) where {P <: ParametrizedPath, S <: AbstractVector, PT <: PTChains}
    n_chains, iterations = size(loss.ptchains)
    T = eltype(t)
    total = zero(T) 
    target_lps = Vector{T}(undef, iterations)
    @inbounds for i in 1:n_chains 
        beta = loss.ptchains.schedule[i]
        # Compute the target log potential
        for j in 1:iterations
            lps = base_potentials(loss.ptchains, i, j)
            @inbounds target_lps[j] = loss.path(t, lps, beta)
        end
        diff = target_lps - @view(loss.ref_lps[:, i])
        w = softmax!(diff)
        total += J_fast(t, loss.path, w, target_lps, loss.ptchains, i)
    end
    return total
end

function (loss::ISSKLLoss{P, Val{false}, PT})(t::S) where {P <: ParametrizedPath, S <: AbstractVector, PT <: PTChains}
    n_chains, iterations = size(loss.ptchains)
    T = eltype(t)
    total = zero(T) 
    target_lps = Vector{T}(undef, iterations)
    @inbounds for i in 1:n_chains 
        beta = loss.ptchains.schedule[i]
        # Compute the target log potential
        for j in 1:iterations
            lps = base_potentials(loss.ptchains, i, j)
            @inbounds target_lps[j] = loss.path(t, lps, beta)
        end
        diff = target_lps - @view(loss.ref_lps[:, i])
        w = softmax!(diff)
        total += J_fast(t, loss.path, w, target_lps, loss.ptchains, i)
    end
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

function _min_ess(loss::ISSKLLoss{P, V, PT}, t) where {P, V, PT}
    n_chains, iterations = size(loss.ptchains)
    lp_buff = Vector{Float64}(undef, iterations)
    min_ess = Inf
    for i in 1:n_chains
        for j in 1:iterations 
            beta = loss.ptchains.schedule[i]
            lps = base_potentials(loss.ptchains, i, j)
            lp_buff[j] = loss.path(t, lps, beta) - loss.ref_lps[j, i]
        end
        this_ess = ess_ratio(lp_buff)
        min_ess = this_ess < min_ess ? this_ess : min_ess
    end
    return min_ess
end


# Newton method with ESS-based backtracking line search.
function adapt_path!(
    problem::PathProblem{<:SamplingProblem, P},
    ptchains::PTChains,
    opt_state::NewtonTrustRegionState,
    ::SKLObjective = SKLObjective(),
) where {P <: ParametrizedPath}
    loss = ISSKLLoss(problem.path, ptchains, false)

    t = extract_param(problem.path)
    l = loss(t)

    prog = Progress(opt_state.max_steps; desc="Newton trust region (autodiff)", offset=5)
    ProgressMeter.update!(prog, 0, force=true, showvalues=[
        ("objective", l),
    ])

    g = DifferentiationInterface.gradient(loss, opt_state.backend, t)
    H = DifferentiationInterface.hessian(loss, opt_state.backend, t)
    for n in 1:opt_state.max_steps

        if nan_grad(g) || any(isnan, H)
            @warn "NaN in gradient or Hessian during Newton trust region update, stopping"
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

        # Backtracking line search: halve α until ESS ≥ δ or α is negligible.
        α     = 1.0
        new_t = nothing
        min_e = 0.0
        while α > 2^-20
            candidate = step!(t + α * Δt, opt_state.prox)
            min_e = _min_ess(loss, candidate)
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
