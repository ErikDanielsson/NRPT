function SKL_loss(
    problem::PathProblem{<:SamplingProblem, <:Path, E},
    ptchains::PTChains{N, T, Val{false}},
) where {E, N, T}
    l = 0.0
    for i in 1:N 
        li = per_chain_loss(problem.path, ptchains, i)
        l += li
    end
    return l
end

function SKL_loss(
    problem::PathProblem{<:SamplingProblem, <:Path, E},
    ptchains::PTChains{N, T, Val{true}},
) where {E, N, T}
    l = tmapreduce(+, 1:N) do i
        per_chain_loss(problem.path, ptchains, i)
    end
    return l
end

# This is the per chain loss
function per_chain_loss(path, ptchains::PTChains{N, V}, i::Int) where {N, V}
    n_chains, iterations = size(ptchains)
    total = 0.0
    if i == 1
        for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = log_potential(path, base_lps, ptchains.schedule[i])
            next_lp = log_potential(path, base_lps, ptchains.schedule[i + 1])
            total += this_lp - next_lp
        end
    elseif i == n_chains
        for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = log_potential(path, base_lps, ptchains.schedule[i])
            prev_lp = log_potential(path, base_lps, ptchains.schedule[i - 1])
            total += this_lp - prev_lp 
        end
    else
        for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = log_potential(path, base_lps, ptchains.schedule[i])
            prev_lp = log_potential(path, base_lps, ptchains.schedule[i - 1])
            next_lp = log_potential(path, base_lps, ptchains.schedule[i + 1])
            total += 2this_lp -prev_lp - next_lp
        end
    end
    return total / iterations
end

function SKL_gradient(
    problem::PathProblem{<:SamplingProblem, <:Path, E},
    ptchains::PTChains{N, T, Val{false}},
) where {E, N, T}
    n_chains, iterations = size(ptchains)
    g_dim = length(extract_param(problem.path))
    g = zeros(g_dim)
    lp_grad_buff = Matrix{Float64}(undef, g_dim, iterations)
    for i in 1:n_chains
        # Compute the straight through gradient and save the log-potentials gradients
        g += loss_gradient(problem.path, ptchains, i, lp_grad_buff)
        # Compute the gradient correction due to the density, use the precomptued log-potential gradients
        g += density_gradient(problem.path, ptchains, i, lp_grad_buff)
    end
    return g
end

function SKL_gradient(
    problem::PathProblem{<:SamplingProblem, <:Path, E},
    ptchains::PTChains{N, T, Val{true}},
) where {E, N, T}
    n_chains, iterations = size(ptchains)
    g_dim = length(extract_param(problem.path))
    g = tmapreduce(+, 1:n_chains) do i
        # Use one buffer per task here, for now...
        lp_grad_buff = Matrix{Float64}(undef, g_dim, iterations)
        # Compute the straight through gradient and save the log-potentials gradients
        gi = loss_gradient(problem.path, ptchains, i, lp_grad_buff)
        # Compute the gradient correction due to the density, use the precomptued log-potential gradients
        gi += density_gradient(problem.path, ptchains, i, lp_grad_buff)  
        return gi
    end
    return g
end

function loss_gradient(path, ptchains::PTChains, i::Int, lp_grad_buff::Matrix{Float64})
    n_chains, iterations = size(ptchains)
    total = zeros(size(lp_grad_buff, 1))
    if i == 1
        g_buff1 = zeros(size(lp_grad_buff, 1)) 
        @views @inbounds for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = gradient!(path, base_lps, ptchains.schedule[i], lp_grad_buff[:, j])
            next_lp = gradient!(path, base_lps, ptchains.schedule[i + 1], g_buff1)
            @. total += this_lp - next_lp
        end
    elseif i == n_chains
        g_buff2 = zeros(size(lp_grad_buff, 1)) 
        @views @inbounds for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = gradient!(path, base_lps, ptchains.schedule[i], lp_grad_buff[:, j])
            prev_lp = gradient!(path, base_lps, ptchains.schedule[i - 1], g_buff2)
            @. total += this_lp - prev_lp 
        end
    else
        g_buff1 = zeros(size(lp_grad_buff, 1)) 
        g_buff2 = zeros(size(lp_grad_buff, 1)) 
        @views @inbounds for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = gradient!(path, base_lps, ptchains.schedule[i], lp_grad_buff[:, j])
            prev_lp = gradient!(path, base_lps, ptchains.schedule[i - 1], g_buff2)
            next_lp = gradient!(path, base_lps, ptchains.schedule[i + 1], g_buff1)
            @. total += 2this_lp - prev_lp - next_lp
        end
    end
    return total / iterations
end

function density_gradient(path, ptchains::PTChains, i::Int, lp_grad_buff::Matrix{Float64})
    n_chains, iterations = size(ptchains)
    lp_grad_buff .-= mean(lp_grad_buff; dims=2)
    total = zeros(size(lp_grad_buff, 1))
    if i == 1
        @views @inbounds for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = log_potential(path, base_lps, ptchains.schedule[i])
            next_lp = log_potential(path, base_lps, ptchains.schedule[i + 1])
            coeff = (this_lp - next_lp)
            @. total += coeff * lp_grad_buff[:, j]
        end
    elseif i == n_chains
        @views @inbounds for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = log_potential(path, base_lps, ptchains.schedule[i])
            prev_lp = log_potential(path, base_lps, ptchains.schedule[i - 1])
            coeff = (this_lp - prev_lp)
            @. total += coeff * lp_grad_buff[:, j]
        end
    else
        @views @inbounds for j in 1:iterations
            base_lps = base_potentials(ptchains, i, j)
            this_lp = log_potential(path, base_lps, ptchains.schedule[i])
            prev_lp = log_potential(path, base_lps, ptchains.schedule[i - 1])
            next_lp = log_potential(path, base_lps, ptchains.schedule[i + 1])
            coeff = (2this_lp -prev_lp - next_lp)
            @. total +=  coeff * lp_grad_buff[:, j]
        end
    end
    return total / (iterations - 1) # This correction makes the covariance estimator unbiased
end