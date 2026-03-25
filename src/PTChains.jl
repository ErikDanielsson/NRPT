mutable struct Indices
    σ::Vector{Int}     # The permutation of state indices, i.e. σ[i] is the chain where state i resides
    σ_inv::Vector{Int} # The permutation of state indices: i.e. σ_inv[j] is the state of chain j
end

Indices(n::Int) = Indices(1:n, 1:n)
Base.copy(is::Indices) = Indices(Base.copy(is.σ), Base.copy(is.σ_inv))

function swap(is::Indices, i, j)
	new_is = copy(is)
	new_is.σ[i] = is.σ[j]
	new_is.σ[j] = is.σ[i]
	new_is.σ_inv[is.σ[i]] = j
	new_is.σ_inv[is.σ[j]] = i
	return new_is
end

function swap!(is::Indices, i, j)
    σ_i = is.σ[i]
    σ_j = is.σ[j]
    is.σ[i] = σ_j
    is.σ[j] = σ_i
    is.σ_inv[σ_i] = j
    is.σ_inv[σ_j] = i
end

mutable struct PTChains{T}
    x::Vector{T}
    chains::Vector{Chain}
    inds::Indices
    iterations::Int
    r_buf::Vector{Float64}
    lps_forward_buf::Vector{Float64}
    lps_backward_buf::Vector{Float64}
end

# function PTChains(starting_inds::Indices, schedule::Vector{Float64}, iterations; dims=2)
#     chains = Vector{Chain}(undef, length(schedule))
#     for (i, beta) in enumerate(schedule)
#         chains[i] = Chain(i, Array{Float64, 2}(undef, dims, iterations), beta)
#     end
#     return PTChains(chains, starting_inds)
# end

function PTChains(x0, init_schedule; dims=2)
    n_chains = length(init_schedule)
    chains = Vector{Chain}(undef, n_chains)
    for (i, beta) in enumerate(init_schedule)
        chains[i] = Chain(i, Array{Float64, 2}(undef, dims, 0), beta)
    end
    inds = Indices(n_chains)
    r_buf          = Vector{Float64}(undef, n_chains - 1)
    lps_forward_buf  = Vector{Float64}(undef, n_chains - 1)
    lps_backward_buf = Vector{Float64}(undef, n_chains - 1)
    return PTChains(x0, chains, inds, 0, r_buf, lps_forward_buf, lps_backward_buf)
end

function refresh_chains!(ptchains::PTChains, schedule, iterations; dims=2)
    ptchains.iterations = iterations
    chains = Vector{Chain}(undef, length(schedule))
    for (i, beta) in enumerate(schedule)
        chains[i] = Chain(i, Array{Float64, 2}(undef, dims, ptchains.iterations), beta)
    end
    ptchains.chains = chains
end

Base.length(chains::PTChains) = length(chains.chains)
Base.size(chains::PTChains) = (length(chains.chains), chains.iterations)

function explore!(problem::PathProblem, chains::PTChains, iteration::Int) where {T}
    Threads.@threads for chain in chains.chains
        state_index = chains.inds.σ[chain.index]
        # println("Iteration $iteration: ch:$(chain.index), st:$(state_index)")
        chains.x[state_index] = explore_chain(problem, chain, chains.x[state_index], iteration)
    end
    # println("x = $(get_state_per_temperature(chains))")
end

function swap_chains(
    problem::PathProblem,
    chains::PTChains,
    iteration::Int,
)
    # println("σ = $(chains.inds.σ)")
    # println("σ_inv = $(chains.inds.σ_inv)")
    # println((iteration % 2 == 1) ? "Odd" : "Even")
    for i in 1:length(chains.chains) - 1
        chain1 = chains.chains[i]
        chain2 = chains.chains[i + 1]
        @assert (chain2.index - chain1.index == 1)
        α, forward, backward = swap_pair(problem, chain1, chain2, iteration)
        # println(α)
        if (i % 2 == iteration % 2)
            A = rand(Bernoulli(α))
            if A
                swap!(chains.inds, i, i + 1)
                # println("Swapping $(i):$(i + 1)")
                # println("σ = $(chains.inds.σ)")
                # println("σ_inv = $(chains.inds.σ_inv)")
            end
        end
        chains.r_buf[i] = 1 - α
        chains.lps_forward_buf[i] = forward
        chains.lps_backward_buf[i] = backward
    end
    return chains.r_buf, chains.lps_forward_buf, chains.lps_backward_buf
end

function get_state_per_temperature(chains::PTChains)
    return chains.x[chains.inds.σ]
end

# The index process describes how the state's shifts over time 
get_index_process(chains::PTChains) = chains.inds.σ_inv
get_inds(chains::PTChains) = chains.inds