function R(
    problem::PathProblem{<:SamplingProblem, P, E},
    n::Int,
    schedule::Vector{Float64},
    lps1::AbstractVector{Float64},
    lps2::AbstractVector{Float64}
) where {P<:ParametrizedPath, E <: Explorer}
    prop1 = log_potential(problem.path, lps1, schedule[n + 1])
    prop2 = log_potential(problem.path, lps2, schedule[n])
    ref2 = log_potential(problem.path, lps1, schedule[n])
    ref1 = log_potential(problem.path, lps2, schedule[n + 1])
    return 1 - exp(min(0, prop1 + prop2 - ref1 - ref2))
end


function T(
    problem::PathProblem{<:SamplingProblem, P, E},
    n::Int,
    schedule::Vector{Float64},
    lps1::AbstractVector{Float64},
    lps2::AbstractVector{Float64}
) where {P<:ParametrizedPath, E <: Explorer}
    prop1 = log_potential(problem.path, lps1, schedule[n + 1])
    prop2 = log_potential(problem.path, lps2, schedule[n])
    ref2 = log_potential(problem.path, lps1, schedule[n])
    ref1 = log_potential(problem.path, lps2, schedule[n + 1])
    return Float64(prop1 + prop2 - ref2 - ref1 > 0)
end

function S(
    problem::PathProblem{<:SamplingProblem, P, E},
    n::Int,
    schedule::Vector{Float64},
    lps1::AbstractVector{Float64},
    lps2::AbstractVector{Float64}
) where {P<:ParametrizedPath, E <: Explorer}
    return (
        gradient(problem.path, lps1, schedule[n])
        + gradient(problem.path, lps2, schedule[n+1])
    )
end

function barrier_grad(
    problem::PathProblem{<:SamplingProblem, P, E},
    chain1::Chain,
    chain2::Chain,
    schedule::Vector{Float64}
) where {P <: ParametrizedPath{<:AbstractArray}, E <: Explorer}
    Ts = [
        T(problem, chain1.index, schedule, lps1, lps2)
        for (lps1, lps2) in
        zip(
            eachcol(chain1.log_potentials),
            eachcol(chain2.log_potentials)
        )
    ]
    Ss = hcat([
        S(problem, chain1.index, schedule, lps1, lps2)
        for (lps1, lps2) in
        zip(
            eachcol(chain1.log_potentials),
            eachcol(chain2.log_potentials)
        )
    ]...)
    return vec(cov(Ss, Ts))
end

function barrier_grad(
    problem::PathProblem{<:SamplingProblem, P, E},
    chain1::Chain,
    chain2::Chain,
    schedule::Vector{Float64}
) where {P <: ParametrizedPath{<:Real}, E <: Explorer}
    Ts = [
        T(problem, chain1.index, schedule, lps1, lps2)
        for (lps1, lps2) in
        zip(
            eachcol(chain1.log_potentials),
            eachcol(chain2.log_potentials)
        )
    ]
    Ss = [
        S(problem, chain1.index, schedule, lps1, lps2)
        for (lps1, lps2) in
        zip(
            eachcol(chain1.log_potentials),
            eachcol(chain2.log_potentials)
        )
    ]
    return cov(Ss, Ts)
end

# Per-pair rejection rate (loss for a single adjacent pair)
function barrier_pair_loss(
    problem::PathProblem{<:SamplingProblem, P, E},
    chain1::Chain,
    chain2::Chain,
    schedule::Vector{Float64}
) where {P <: ParametrizedPath, E <: Explorer}
    rs = [
        R(problem, chain1.index, schedule, lps1, lps2)
        for (lps1, lps2) in zip(
            eachcol(chain1.log_potentials),
            eachcol(chain2.log_potentials)
        )
    ]
    loss = mean(rs)
    return loss
end

# Aggregate loss over all adjacent chain pairs
function barrier_loss(
    problem::PathProblem{<:SamplingProblem, <:ParametrizedPath, E},
    ptchains::PTChains,
    schedule::Vector{Float64}
) where {E}
    chains = ptchains.chains
    rejs = [
        barrier_pair_loss(problem, chain1, chain2, schedule)
        for (chain1, chain2) in zip(chains[1:end-1], chains[2:end])
    ]
    return sum(rejs)
end

# Aggregate gradient over all adjacent chain pairs
function barrier_gradient(
    problem::PathProblem{<:SamplingProblem, <:ParametrizedPath, E},
    ptchains::PTChains,
    schedule::Vector{Float64}
) where {E}
    chains = ptchains.chains
    g =  sum(
        barrier_grad(problem, chain1, chain2, schedule)
        for (chain1, chain2) in zip(chains[1:end-1], chains[2:end])
    )
    return g
end