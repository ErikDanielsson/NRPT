# Log IS weights for one chain.
# ref_lps[i] = log_potential(φ₀, lps_i, β) stored before optimization starts.
# log w_i = log π_{φ,β}(e_i) - log π_{φ₀,β}(e_i)
#          = log_potential(φ, lps_i, β) - ref_lps[i]
function chain_log_weights(
    problem::PathProblem,
    chain::Chain,
    schedule::Vector{Float64},
    ref_lps::Vector{Float64},
)
    β = schedule[chain.index]
    return [
        log_potential(problem.path, lps, β) - ref_lps[i]
        for (i, lps) in enumerate(eachcol(chain.log_potentials))
    ]
end

# ESS ratio: (Σ wᵢ)² / (n Σ wᵢ²), computed in log space for stability.
# Equals 1 when all weights are equal, approaches 0 when one weight dominates.
function ess_ratio(log_weights::Vector{Float64})
    n   = length(log_weights)
    ls  = logsumexp(log_weights)
    ls2 = logsumexp(2 .* log_weights)
    return exp(2ls - log(n) - ls2)
end

# IS-weighted SKL loss for one chain: L_n(φ) = Σᵢ w̃ᵢ J(φ, βₙ, eᵢ)
function IS_SKL_loss_chain(
    problem::PathProblem{<:SamplingProblem, <:Path, E},
    chain::Chain,
    schedule::Vector{Float64},
    ref_lps::Vector{Float64},
) where {E}
    log_w = chain_log_weights(problem, chain, schedule, ref_lps)
    w̃ = softmax(log_w)
    Js = [J(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]
    return dot(w̃, Js)
end

# IS-weighted SKL gradient - scalar path parameter.
# ∇L_n(φ) = E_{w̃}[∇J] + Cov_{w̃}(∇W, J)
function IS_SKL_grad_chain(
    problem::PathProblem{<:SamplingProblem, P, E},
    chain::Chain,
    schedule::Vector{Float64},
    ref_lps::Vector{Float64},
) where {P <: ParametrizedPath{<:Real}, E}
    log_w = chain_log_weights(problem, chain, schedule, ref_lps)
    w̃ = softmax(log_w)

    Js  = [J( problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]
    ∇Ws = [∇W(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]
    ∇Js = [∇J(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]

    J_mean = dot(w̃, Js)
    W_mean = dot(w̃, ∇Ws)
    g1 = dot(w̃, ∇Ws .* Js) - W_mean * J_mean  # IS-weighted Cov(∇W, J)
    g2 = dot(w̃, ∇Js)                            # IS-weighted E[∇J]
    return g1 + g2
end

# IS-weighted SKL gradient - vector path parameter.
function IS_SKL_grad_chain(
    problem::PathProblem{<:SamplingProblem, P, E},
    chain::Chain,
    schedule::Vector{Float64},
    ref_lps::Vector{Float64},
) where {P <: ParametrizedPath{<:AbstractArray}, E}
    log_w = chain_log_weights(problem, chain, schedule, ref_lps)
    w̃ = softmax(log_w)

    Js  = [J( problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]
    ∇Ws = hcat([∇W(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]...)  # d × n
    ∇Js = hcat([∇J(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]...)  # d × n

    J_mean = dot(w̃, Js)
    W_mean = ∇Ws * w̃                                    # d-vector
    g1 = vec(∇Ws * (w̃ .* Js) - W_mean * J_mean)         # IS-weighted Cov(∇W, J)
    g2 = vec(∇Js * w̃)                                    # IS-weighted E[∇J]
    return g1 + g2
end
