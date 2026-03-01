# If we do not have an optimizer do nothing
function adapt_path!(problem, x::Matrix{T}, schedule, opt_state::NoOptState) where {T}
    return sum(mcmc_loss(problem.path, x, n, schedule) for n in eachindex(schedule))
end

# If we have a static path do nothing
function adapt_path!(problem::PathProblem{StaticPath, E}, x::Matrix{T}, schedule, opt_state::ProximalStochOptState) where {T, E}
end

# function adapt_path!(
#     problem::PathProblem{<:ParametrizedPath, E},
#     x::Matrix{T},
#     schedule,
#     opt_state::ProximalStochOptState{S, Pr}
# ) where {T, P, E, S, Pr}
#     l = sum(mcmc_loss(problem.path, x, n, schedule) for n in eachindex(schedule))
#     g = sum([mcmc_grad(problem.path, x, n, schedule) for n in eachindex(schedule)])
#     problem.path.params = step!(problem.path.params, g, opt_state)
#     return l
# end

function adapt_path!(
    problem::PathProblem{<:ParametrizedPath, E},
    x::Matrix{T},
    schedule,
    opt_state::ProximalStochOptState{S, Pr}
) where {T, E, S, Pr}
    all_inds = eachindex(schedule)
    chunks = Iterators.partition(all_inds, cld(length(all_inds), Threads.nthreads()))
    tasks = map(chunks) do chunk
        Threads.@spawn acc_grad(chunk, problem.path, x, schedule)
    end
    partial_grads = fetch.(tasks)
    l, g = sum(partial_grads)
    new_param = step!(extract_param(problem.path), g, opt_state)
    set_param!(problem.path, new_param)
    return l
end

function acc_grad(inds, path::P, x::Matrix{T}, schedule) where {P<:ParametrizedPath, T}
    l = sum(mcmc_loss(path, x, n, schedule) for n in inds)
    g = sum([mcmc_grad(path, x, n, schedule) for n in eachindex(schedule)])
    return [l, g]
end

function ∇J(path::ParametrizedPath, n, schedule, x::T) where {T}
    if n == 1
        return gradient(path, x, schedule[1]) - gradient(path, x, schedule[2])
    elseif n == length(schedule)
        return gradient(path, x, schedule[end]) - gradient(path, x, schedule[end-1])
    else
        return (
            2gradient(path, x, schedule[n])
            - gradient(path, x, schedule[n+1])
            - gradient(path, x, schedule[n-1])
        )
    end
end

function J(path::ParametrizedPath, n, schedule, x::T) where {T}
    if n == 1
        return log_potential(path, x, schedule[1]) - log_potential(path, x, schedule[2])
    elseif n == length(schedule)
        return log_potential(path, x, schedule[end]) - log_potential(path, x, schedule[end-1])
    else
        return (
            2log_potential(path, x, schedule[n])
            - log_potential(path, x, schedule[n+1])
            - log_potential(path, x, schedule[n-1])
        )
    end
end

function ∇W(path::ParametrizedPath, n, schedule, x::T) where {T}
    return gradient(path, x, schedule[n]) 
end

function mcmc_loss(path::ParametrizedPath, x::Matrix{T}, n, schedule::Vector{Float64}) where {T}
    samples = x[n, :]
    return mean(J(path, n, schedule, s) for s in samples)
end

function mcmc_grad(path::ParametrizedPath{<:Real}, x::Matrix{T}, n, schedule::Vector{Float64}) where {T}
    samples = x[n, :]
    g1s = [∇W(path, n, schedule, s) for s in samples]
    g1 = cov(g1s, [J(path, n, schedule, s) for s in samples])
    g2 = mean(∇J(path, n, schedule, s) for s in samples)
    return g1 + g2
end

function mcmc_grad(path::ParametrizedPath{<:AbstractArray}, x::Matrix{T}, n, schedule::Vector{Float64}) where {T}
    samples = x[n, :]
    g1s = hcat([∇W(path, n, schedule, s) for s in samples]...)
    g1 = cov(g1s', [J(path, n, schedule, s) for s in samples])
    g2 = mean(∇J(path, n, schedule, s) for s in samples)
    return vec(g1 + g2)
end