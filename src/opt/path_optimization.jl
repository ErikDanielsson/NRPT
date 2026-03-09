# If we do not have an optimizer do nothing
function adapt_path!(problem, ptchains::PTChains, schedule, opt_state::NoOptState)
    return sum(mcmc_loss(problem, chain, schedule) for chain in ptchains.chains)
end

# If we have a static path do nothing
function adapt_path!(problem::PathProblem{<:SamplingProblem, StaticPath, E}, ptchains::PTChains, schedule, opt_state::ProximalStochOptState) where {E}
end

function adapt_path!(
    problem::PathProblem{<:SamplingProblem, <:ParametrizedPath, E},
    ptchains::PTChains,
    schedule,
    opt_state::ProximalStochOptState{S, Pr}
) where {E, S, Pr}
    chains = ptchains.chains 
    chunks = Iterators.partition(chains, cld(length(chains), Threads.nthreads()))
    tasks = map(chunks) do chunk
        Threads.@spawn acc_grad(problem, chunk, schedule)
    end
    partial_grads = fetch.(tasks)
    l, g = sum(partial_grads)
    new_param = step!(extract_param(problem.path), g, opt_state)
    set_param!(problem.path, new_param)
    return l
end

function acc_grad(problem::PathProblem{<:SamplingProblem, P, E}, chains::AbstractVector{Chain}, schedule) where {P<:ParametrizedPath, E <: Explorer}
    l = sum(mcmc_loss(problem, chain, schedule) for chain in chains)
    g = sum([mcmc_grad(problem, chain, schedule) for chain in chains])
    return [l, g]
end

function ∇J(problem::PathProblem{<:SamplingProblem, P, E}, n, schedule, log_potential::AbstractVector{Float64}) where {P<:ParametrizedPath, E <: Explorer}
    if n == 1
        return gradient(problem.path, log_potential, schedule[1]) - gradient(problem.path, log_potential, schedule[2])
    elseif n == length(schedule)
        return gradient(problem.path, log_potential, schedule[end]) - gradient(problem.path, log_potential, schedule[end-1])
    else
        return (
            2gradient(problem.path, log_potential, schedule[n])
            - gradient(problem.path, log_potential, schedule[n+1])
            - gradient(problem.path, log_potential, schedule[n-1])
        )
    end
end

function J(problem::PathProblem{<:SamplingProblem, P, E}, n::Int, schedule::Vector{Float64}, lps::AbstractVector{Float64}) where {P<:ParametrizedPath, E <: Explorer}
    if n == 1
        return log_potential(problem.path, lps, schedule[1]) - log_potential(problem.path, lps, schedule[2])
    elseif n == length(schedule)
        return log_potential(problem.path, lps, schedule[end]) - log_potential(problem.path, lps, schedule[end-1])
    else
        return (
            2log_potential(problem.path, lps, schedule[n])
            - log_potential(problem.path, lps, schedule[n+1])
            - log_potential(problem.path, lps, schedule[n-1])
        )
    end
end

function ∇W(problem::PathProblem{<:SamplingProblem, P, E}, n, schedule, lps::AbstractVector{Float64}) where {P<:ParametrizedPath, E <: Explorer}
    return gradient(problem.path, lps, schedule[n]) 
end

function mcmc_loss(problem::PathProblem{<:SamplingProblem, P, E}, chain::Chain, schedule::Vector{Float64}) where {P<:ParametrizedPath, E <: Explorer}
    return mean(J(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials))
end

function mcmc_grad(problem::PathProblem{<:SamplingProblem, P, E}, chain::Chain, schedule::Vector{Float64}) where {P <: ParametrizedPath{<:AbstractArray}, E <: Explorer}
    g1s = hcat([∇W(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]...)
    g1 = cov(g1s', [J(problem, chain.index, schedule, lps) for lps in  eachcol(chain.log_potentials)])
    g2 = mean(∇J(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials))
    return vec(g1 + g2)
end

function mcmc_grad(problem::PathProblem{<:SamplingProblem, P, E}, chain::Chain, schedule::Vector{Float64}) where {P <: ParametrizedPath{<:Real}, E <: Explorer}
    g1s = [∇W(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials)]
    g1 = cov(vec(g1s), [J(problem, chain.index, schedule, lps) for lps in  eachcol(chain.log_potentials)])
    g2 = mean(∇J(problem, chain.index, schedule, lps) for lps in eachcol(chain.log_potentials))
    return g1 + g2
end
