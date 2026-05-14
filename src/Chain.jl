mutable struct Chain{T}
    x::T
    lp_buff::Vector{Float64}
end

function refresh_chain!(chain::Chain, schedule::Vector{Float64})
    return chain.lp_buff .= NaN
end

function refresh_chain!(chain::Chain, schedule::Vector{Float64}, iterations::Int; dims = 2)
    return chain.lp_buff .= NaN
end

function write_state(sink::Ref{T}, chain::Chain{T}) where {T}
    return sink[] = copy(chain.x)
end

function write_state(sink::Ref{T}, chain::Chain{T}) where {T <: AbstractVector}
    sink[] = copy(chain.x)
    return
end

function record_base_potentials!(problem::PathProblem, chain::Chain, x, iteration::Int)
    return @inbounds chain.base_potentials[:, iteration] = base_potentials!(problem, x, chain.lp_buff)
end

function base_potentials(chain::Chain, iteration::Int)
    return @view(chain.base_potentials[:, iteration])
end

function explore_chain!(problem::PathProblem, chain::Chain, beta::Float64, base_lp_loc::S) where {S <: AbstractVector{<:Real}}

    # Explore at the current temperature
    step!(problem, chain.x, beta, chain.lp_buff)

    # Record the base potentials V0 & V1 at the current temperature
    base_potentials!(problem, chain.x, base_lp_loc)
    return
end

function compute_pair_swap_old(problem::PathProblem, chain1::Chain, chain2::Chain, iteration::Int)
    base_potentials1 = base_potentials(chain1, iteration)
    base_potentials2 = base_potentials(chain2, iteration)
    # The proposal probabilities is the path evaluated in the swapped values
    # and the reference is the path evaluated in the current values
    lp_prop1 = log_potential(problem.path, base_potentials1, chain2.beta)
    lp_prop2 = log_potential(problem.path, base_potentials2, chain1.beta)
    lp_ref1 = log_potential(problem.path, base_potentials1, chain1.beta)
    lp_ref2 = log_potential(problem.path, base_potentials2, chain2.beta)

    α = exp(min(0, lp_prop1 + lp_prop2 - lp_ref1 - lp_ref2))
    α = isnan(α) ? zero(α) : α

    lps_forward = lp_prop1 - lp_ref1
    lps_backward = lp_prop2 - lp_ref2
    return α, lps_forward, lps_backward
end
