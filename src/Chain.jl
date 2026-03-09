mutable struct Chain
    index::Int
    log_potentials::AbstractArray{Float64, 2}
    beta::Float64
end

function record_log_potential!(problem::PathProblem, chain::Chain, x, iteration::Int)
    chain.log_potentials[:, iteration] = log_potentials(problem, x)
end

function log_potentials(chain::Chain, iteration::Int)
    return chain.log_potentials[:, iteration]
end

function explore_chain(problem::PathProblem, chain::Chain, x::T, iteration::Int) where {T}
    x = step(problem, x, chain.beta)
    record_log_potential!(problem, chain, x, iteration)
    return x
end

function swap_pair(problem::PathProblem, chain1::Chain, chain2::Chain, iteration::Int)
    log_potentials1 = log_potentials(chain1, iteration)
    log_potentials2 = log_potentials(chain2, iteration)
    # The proposal probabilities is the path evaluated in the swapped values
    # and the reference is the path evaluated in the current values
    lp_prop1 = log_potential(problem.path, log_potentials1, chain2.beta)
    lp_prop2 = log_potential(problem.path, log_potentials2, chain1.beta)
    lp_ref1 = log_potential(problem.path, log_potentials1, chain1.beta)
    lp_ref2 = log_potential(problem.path, log_potentials2, chain2.beta)
    α = exp(min(0, lp_prop1 + lp_prop2 - lp_ref1 - lp_ref2))
    lps_forward = lp_prop1 - lp_ref1
    lps_backward = lp_prop2 - lp_ref2
    return α, lps_forward, lps_backward
end