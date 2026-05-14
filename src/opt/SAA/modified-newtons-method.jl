struct NewtonTrustRegionState{T, Pr <: ProximalState, E <: ESSCriterion} <: Optimizer
    prox::Pr
    backend::AbstractADType
    max_steps::Int      # hard cap on inner Newton iterations per round
    λ_reg::Float64      # Tikhonov regularisation floor added to the Hessian diagonal
    n_steps::Vector{Int}
    min_eigvals::Vector{Float64}  # minimum eigenvalue of H each step (negative = non-convex)
    xs::T
    crit::E
end

const DEFAULT_λ_reg = 1.0e-3

NewtonTrustRegionState(backend::AbstractADType; δ = 0.9, max_steps = 20, λ_reg = DEFAULT_λ_reg, prox = NoProx()) =
    NewtonTrustRegionState(prox, backend, max_steps, Float64(λ_reg), Int[], Float64[], [], FixedrESSCriterion(δ))

NewtonTrustRegionState(backend::AbstractADType, crit::ESSCriterion; max_steps = 20, λ_reg = DEFAULT_λ_reg, prox = NoProx()) =
    NewtonTrustRegionState(prox, backend, max_steps, Float64(λ_reg), Int[], Float64[], [], crit)

function init(problem::PathProblem{<:SamplingProblem, <:ParametrizedPath}, state::NewtonTrustRegionState)
    return NewtonTrustRegionState(state.prox, state.backend, state.max_steps, state.λ_reg, Int[], Float64[], [extract_param(problem.path)], state.crit)
end

get_last_eta(ntrs::NewtonTrustRegionState) = length(ntrs.min_eigvals) > 0 ? ntrs.min_eigvals[end] : nothing


# Newton method with ESS-based backtracking line search.
function opt_modified_newton_trust_region(
        problem::PathProblem{<:SamplingProblem, P},
        opt_state::NewtonTrustRegionState,
        loss::SNISSKLLoss,
        rESS_lb::Float64,
        progress::Bool
    ) where {P <: ParametrizedPath}

    t = extract_param(problem.path)
    l = loss(t)

    prog = Progress(opt_state.max_steps; desc = "Newton trust region (autodiff)", offset = 7, enabled = progress)
    ProgressMeter.update!(
        prog, 0, force = true, showvalues = [
            ("objective", l),
        ]
    )

    g = DifferentiationInterface.gradient(loss, opt_state.backend, t)
    H = DifferentiationInterface.hessian(loss, opt_state.backend, t)
    min_eig = minimum(eigvals(Symmetric(H)))

    for n in 1:opt_state.max_steps
        if nan_grad(g) || any(isnan, H)
            @warn "NaN in gradient or Hessian during Newton trust region update, stopping"
            @warn "Last parameter value $t"
            @warn "Minimal eig $min_eig"
            break
        end

        min_eig = minimum(eigvals(Symmetric(H)))
        push!(opt_state.min_eigvals, min_eig)

        Δt = _newton_step(g, H, min_eig, opt_state.λ_reg)
        if sqrt(norm2(Δt)) < 1.0e-12
            ProgressMeter.update!(
                prog, force = true, showvalues = [
                    ("objective", l),
                    ("Δt", Δt),
                ]
            )
            return l
        end

        # Backtracking line search: halve α until relative ESS > δ and we improve the objective
        α = 1.0
        new_t = nothing
        min_e = 0.0
        while α > 2^-10
            candidate = step!(t + α * Δt, opt_state.prox)
            l_cand = loss(candidate)
            min_e = min_ess(loss, candidate)
            if (min_e >= rESS_lb) && (l_cand <= l)
                new_t = candidate
                break
            end
            α /= 2
        end

        if new_t === nothing
            push!(opt_state.n_steps, n)
            next!(
                prog, showvalues = [
                    ("objective", l),
                    ("rESS (bound)", "$min_e ($rESS_lb)"),
                    ("α", α),
                    ("||g||", sqrt(norm2(g))),
                    ("λ_min(H)", min_eig),
                ]
            )
            push!(opt_state.xs, t)
            return l
        end

        set_param!(problem.path, new_t)
        l = loss(new_t)

        next!(
            prog, showvalues = [
                ("objective", l),
                ("rESS (bound)", "$min_e ($rESS_lb)"),
                ("α", α),
                ("||g||", sqrt(norm2(g))),
                ("λ_min(H)", min_eig),
            ]
        )
        t = extract_param(problem.path)
        g = DifferentiationInterface.gradient!(loss, g, opt_state.backend, t)
        H = DifferentiationInterface.hessian!(loss, H, opt_state.backend, t)
    end

    push!(opt_state.n_steps, opt_state.max_steps)
    push!(opt_state.xs, t)
    return l
end

# Modified Newton step to guarantee descent direction
function _newton_step(g::Vector{Float64}, H::Matrix{Float64}, min_eig::Float64, λ_reg::Float64)
    λ = max(0.0, -min_eig) + λ_reg
    return (Symmetric(H) + λ * I) \ (-g)
end