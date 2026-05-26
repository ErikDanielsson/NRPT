# This file implements the SNIS estimator for the SKL loss
# Gradients of the loss can be computed via autodiff

struct SNISSKLLoss{P <: ParametrizedPath, Tr <: Val, PT <: PTChains}
    path::P
    ptchains::PT
    ref_lps::Matrix{Float64}
end

function SNISSKLLoss(path::P, ptchains::PT, threaded::Bool) where {P <: ParametrizedPath, PT <: PTChains}
    n_chains, iterations = size(ptchains)
    # Compute the log potential at ϕ_0
    ref_lps = Matrix{Float64}(undef, iterations, n_chains)
    for i in 1:n_chains
        β = ptchains.schedule[i]
        for j in 1:iterations
            ref_lps[j, i] = log_potential(path, base_potentials(ptchains, i, j), β)
        end
    end
    return SNISSKLLoss{P, Val{threaded}, PT}(path, ptchains, ref_lps)
end

# Single threaded implementation
function (loss::SNISSKLLoss{P, Val{false}, PT})(t::S) where {P <: ParametrizedPath, S <: AbstractVector, PT <: PTChains}
    n_chains, iterations = size(loss.ptchains)
    T = eltype(t)
    total = zero(T)
    target_lps = Vector{T}(undef, iterations)
    @inbounds for i in 1:n_chains
        beta = loss.ptchains.schedule[i]
        # Compute the target log potential
        for j in 1:iterations
            lps = base_potentials(loss.ptchains, i, j)
            @inbounds target_lps[j] = loss.path(t, lps, beta)
        end
        diff = target_lps - @view(loss.ref_lps[:, i])
        w = softmax!(diff)
        total += J_fast(t, loss.path, w, target_lps, loss.ptchains, i)
    end
    return total
end

# Multithreaded implementation
function (loss::SNISSKLLoss{P, Val{true}, PT})(t::S) where {P <: ParametrizedPath, S <: AbstractVector, PT <: PTChains}
    n_chains, iterations = size(loss.ptchains)
    T = eltype(t)
    total = tmapreduce(+, 1:n_chains; scheduler = :static, init = zero(T)) do i
        target_lps = Vector{T}(undef, iterations)
        beta = loss.ptchains.schedule[i]
        # Compute the target log potential
        for j in 1:iterations
            lps = base_potentials(loss.ptchains, i, j)
            @inbounds target_lps[j] = loss.path(t, lps, beta)
        end
        diff = target_lps - @view(loss.ref_lps[:, i])
        w = softmax!(diff)
        return J_fast(t, loss.path, w, target_lps, loss.ptchains, i)
    end
    return total
end

function J_fast(t, path, w, target_lps, ptchains::PTChains{N, V}, i::Int) where {N, V}
    total = zero(eltype(t))
    if i == 1
        @views @inbounds for j in eachindex(w, target_lps)
            base_lps = base_potentials(ptchains, i, j)
            total += w[j] * (target_lps[j] - path(t, base_lps, ptchains.schedule[i + 1]))
        end
        return total
    elseif i == N
        @views @inbounds for j in eachindex(w, target_lps)
            base_lps = base_potentials(ptchains, i, j)
            total += w[j] * (target_lps[j] - path(t, base_lps, ptchains.schedule[i - 1]))
        end
        return total
    else
        @views @inbounds for j in eachindex(w, target_lps)
            base_lps = base_potentials(ptchains, i, j)
            lp_pos = path(t, base_lps, ptchains.schedule[i + 1])
            lp_neg = path(t, base_lps, ptchains.schedule[i - 1])
            total += w[j] * (2target_lps[j] - lp_pos - lp_neg)
        end
        return total
    end
end


# Minimum ESS ratio across all chains at a candidate parameter value t.
# function min_ess(path, t, chains, schedule, ref_lps)
#     return minimum(
#         ess_ratio(
#                 [
#                     path(t, lps, schedule[chain.index]) - ref[i]
#                     for (i, lps) in enumerate(eachcol(chain.base_potentials))
#                 ]
#             )
#             for (chain, ref) in zip(chains, ref_lps)
#     )
# end

function relative_ess(log_weights::Vector{Float64})
    n = length(log_weights)
    ls = logsumexp_(log_weights)
    ls2 = logsumexp_(2 .* log_weights)
    return exp(2ls - log(n) - ls2)
end

function min_ess(loss::SNISSKLLoss{P, V, PT}, t) where {P, V, PT}
    n_chains, iterations = size(loss.ptchains)
    lp_buff = Vector{Float64}(undef, iterations)
    min_ress = Inf
    for i in 1:n_chains
        for j in 1:iterations
            beta = loss.ptchains.schedule[i]
            lps = base_potentials(loss.ptchains, i, j)
            lp_t = loss.path(t, lps, beta)
            lp_buff[j] = lp_t - loss.ref_lps[j, i]
        end
        this_ress = relative_ess(lp_buff)
        min_ress = this_ress < min_ress ? this_ress : min_ress
    end
    return min_ress
end