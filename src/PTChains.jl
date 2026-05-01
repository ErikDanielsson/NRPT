mutable struct Indices
    i::Vector{Int}     # The permutation of state indices, i.e. σ[i] is the chain where state i resides
    j::Vector{Int} # The permutation of state indices: i.e. σ_inv[j] is the state of chain j
end

Indices(n::Int) = Indices(1:n, 1:n)
Base.copy(is::Indices) = Indices(Base.copy(is.σ), Base.copy(is.σ_inv))

function swap!(is::Indices, i1)
    i2 = i1 + 1
    @inbounds j1 = is.j[i1]
    @inbounds j2 = is.j[i2]
    @inbounds is.j[i1] = j2
    @inbounds is.j[i2] = j1
    @inbounds is.i[j1] = i2
    @inbounds is.i[j2] = i1
    return 
end

function invariance(inds::Indices)
    @assert all(inds.i[inds.j] .== inds.j[inds.i])
    return @assert all(inds.i[inds.j] .== 1:length(inds.i))
end

mutable struct PTChains{N, T, Tr <: Val}
    chains::SVector{N, Chain{T}}
    inds::Indices
    schedule::Vector{Float64}
    iterations::Int
    r_buf::Vector{Float64}
    lps_forward_buf::Vector{Float64}
    lps_backward_buf::Vector{Float64}
    base_lps::Array{Float64, 3} # 2 \times iterations \times n_chains
end

@inline function base_potentials(ptchains::PTChains, i, iteration)
    @inbounds return @view(ptchains.base_lps[:, iteration, i])
end

function PTChains(x0::Vector{T}, init_schedule; threaded = true, dims = 2) where {T}
    n_chains = length(init_schedule)
    mut_chains = Vector{Chain}(undef, n_chains)
    for i in eachindex(mut_chains, x0)
        mut_chains[i] = Chain{T}(x0[i], Vector{Float64}(undef, dims))
    end
    chains = SVector{n_chains, Chain}(mut_chains)
    inds = Indices(n_chains)
    r_buf = Vector{Float64}(undef, n_chains - 1)
    lps_forward_buf = Vector{Float64}(undef, n_chains - 1)
    lps_backward_buf = Vector{Float64}(undef, n_chains - 1)
    base_lps = Array{Float64, 3}(undef, 2, 0, n_chains)
    return PTChains{n_chains, T, Val{threaded}}(
        chains,
        inds,
        init_schedule,
        0,
        r_buf,
        lps_forward_buf,
        lps_backward_buf,
        base_lps
    )
end

function refresh_chains!(ptchains::PTChains{N, T, Tr}, schedule, iterations; dims = 2) where {N, T, Tr}
    ptchains.schedule .= schedule
    if iterations != ptchains.iterations
        ptchains.base_lps = Array{Float64, 3}(undef, dims, iterations, N) 
        ptchains.iterations = iterations
    end
    return
end

Base.length(::PTChains{N, T, Tr}) where {N, T, Tr} = N
Base.size(chains::PTChains{N, T, Tr}) where {N, T, Tr} = (N, chains.iterations)

@inline function get_beta_chain_j(ptchains::PTChains, j::Int, iteration::Int)
    @inbounds i = ptchains.inds.i[j]
    @inbounds beta = ptchains.schedule[i]
    @inbounds chain = ptchains.chains[j]
    base_lp_loc = base_potentials(ptchains, i, iteration)
    return beta, chain, base_lp_loc
end


function explore!(problem::PathProblem, ptchains::PTChains{N, T, Val{true}}, iteration::Int) where {N, T}
    tforeach(1:N; scheduler = :static) do j
        beta, chain, base_lp_loc = get_beta_chain_j(ptchains, j, iteration)
        explore_chain!(problem, chain, beta, base_lp_loc)  # Explore on machine j
    end
    return 
end

function explore!(problem::PathProblem, ptchains::PTChains{N, T, Val{false}}, iteration::Int) where {N, T}
    for j in 1:N
        beta, chain, base_lp_loc = get_beta_chain_j(ptchains, j, iteration)
        explore_chain!(problem, chain, beta, base_lp_loc)  # Explore on machine j
    end
    return
end

@inline function compute_pair_swap(problem::PathProblem, ptchains::PTChains, i::Int, iteration::Int)
    base_potentials1 = base_potentials(ptchains, i, iteration)
    base_potentials2 = base_potentials(ptchains, i + 1, iteration)
    @inbounds beta1 = ptchains.schedule[i]
    @inbounds beta2 = ptchains.schedule[i + 1]
    # The proposal probabilities is the path evaluated in the swapped values
    # and the reference is the path evaluated in the current values
    lp_prop1 = log_potential(problem.path, base_potentials1, beta2)
    lp_prop2 = log_potential(problem.path, base_potentials2, beta1)
    lp_ref1 = log_potential(problem.path, base_potentials1, beta1)
    lp_ref2 = log_potential(problem.path, base_potentials2, beta2)

    α = exp(min(0, lp_prop1 + lp_prop2 - lp_ref1 - lp_ref2))
    α = isnan(α) ? zero(α) : α

    lps_forward = lp_prop1 - lp_ref1
    lps_backward = lp_prop2 - lp_ref2
    return α, lps_forward, lps_backward
end


function swap_chains(
        problem::PathProblem,
        ptchains::PTChains{N, T, Tr},
        iteration::Int,
    ) where {N, T, Tr}
    for i in 1:(N - 1) # Iterate in temperature order
        α, forward, backward = compute_pair_swap(problem, ptchains, i, iteration)
        if (i % 2 == iteration % 2)
            A = rand() <= α
            if A
                swap!(ptchains.inds, i)
            end
        end
        @inbounds ptchains.r_buf[i] = 1 - α
        @inbounds ptchains.lps_forward_buf[i] = forward
        @inbounds ptchains.lps_backward_buf[i] = backward
    end
    return ptchains.r_buf, ptchains.lps_forward_buf, ptchains.lps_backward_buf
end

function set_state_per_temperature!(ptchains::PTChains{N, T, Tr}, sink_arr::S) where {N, T, Tr, S <: AbstractVector{T}}
    for i in 1:N
        sink = Ref(sink_arr, i)
        j = ptchains.inds.j[i]
        chain = ptchains.chains[j]
        write_state(sink, chain)
    end
    return
end

# The index process describes how the state's shifts over time
get_index_process(chains::PTChains) = chains.inds.i
get_inds(chains::PTChains) = chains.inds
