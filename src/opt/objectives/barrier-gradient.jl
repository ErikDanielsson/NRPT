function T(
    problem::PathProblem{<:SamplingProblem, P, E},
    n::Int,
    schedule::Vector{Float64},
    lps1::AbstractVector{Float64},
    lps2::AbstractVector{Float64}
) where {P<:ParametrizedPath, E <: Explorer}
    prop1 = log_potential(problem.path, lps1, schedule[n])
    prop2 = log_potential(problem.path, lps2, schedule[n + 1])
    ref2 = log_potential(problem.path, lps1, schedule[n + 1])
    ref1 = log_potential(problem.path, lps2, schedule[n])
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