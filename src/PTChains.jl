mutable struct Indices
    σ::Vector{Int}
    σ_inv::Vector{Int}
end

Indices(n::Int) = Indices(1:n, 1:n)
copy(is::Indices) = Indices(Base.copy(is.σ), Base.copy(is.σ_inv))

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

mutable struct PTChains
    chains::Vector{Chain}
    inds::Indices
end

function PTChains(starting_inds::Indices, schedule::Vector{Float64}, iterations; dims=2)
    chains = Vector{Chain}(undef, length(schedule))
    for (i, beta) in enumerate(schedule)
        chains[i] = Chain(i, Array{Float64, 2}(undef, dims, iterations), beta)
    end
    return PTChains(chains, starting_inds)
end

Base.length(chains::PTChains) = length(chains.chains)

function explore(problem::PathProblem, chains::PTChains, x::Vector{T}, iteration::Int) where {T}
    for chain in chains.chains
        state_index = chains.inds.σ[chain.index]
        x[state_index] = explore_chain(problem, chain, x[state_index], iteration)
    end
    return x
end

function swap_chains(
    problem::PathProblem,
    chains::PTChains,
    iteration::Int,
)
    r = Vector{Float64}(undef, length(chains) - 1)
    lps_forward = Vector{Float64}(undef, length(chains) - 1)
    lps_backward = Vector{Float64}(undef, length(chains) - 1)
    for (i, (chain1, chain2)) in enumerate(zip(chains.chains[1:end-1], chains.chains[2:end]))
        @assert (chain2.index - chain1.index == 1)
        α, forward, backward = swap_pair(problem, chain1, chain2, iteration)
        if (i % 2 == iteration % 2) 
            A = rand(Bernoulli(α))
            if A
                swap!(chains.inds, i, i + 1)
            end
        end
        r[i] = 1 - α
        lps_forward[i] = forward
        lps_backward[i] = backward
    end
    return r, lps_forward, lps_backward
end

function get_state_per_temperature(chains::PTChains, x::Vector{T}) where {T}
    return x[chains.inds.σ]
end

get_index_process(chains::PTChains) = chains.inds.σ
get_inds(chains::PTChains) = chains.inds