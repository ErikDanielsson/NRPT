# Perform the deterministic even-odd scheme
function DEO_naive(
	x::Vector{T},
	inds::Indices,
	iterations::Int,
	schedule::Vector{Float64},
	problem::PTProblem
) where T
    n_chains = length(schedule)
	# Accumulators
    xs = Matrix{T}(undef, iterations, n_chains)
    r = Matrix{Float64}(undef, iterations, n_chains - 1)
	index_process = Matrix{Int}(undef, n_chains, iterations)
	norm_const = Vector{T}(undef, iterations)
    for n in 1:iterations
		# Exploration
        Threads.@threads for i in 1:n_chains
			x[inds.σ[i]] = problem.explorer(x[inds.σ[i]], schedule[i])
        end
		# Communication
        for i in 1:(n_chains - 1)
            α = exp(min(0, 
                -(problem.V(x[inds.σ[i]], schedule[i+1])
				+ problem.V(x[inds.σ[i+1]], schedule[i])
				- problem.V(x[inds.σ[i]], schedule[i])
				- problem.V(x[inds.σ[i+1]], schedule[i+1]))
            ))
            r[i, n] = (1 - α)
            A = rand(Bernoulli(α))
            if (i % 2 == n % 2) && A
				inds = swap(inds, i, i + 1)
            end
        end
		xs[n, inds.σ_inv] = x_curr
		index_process[n, inds.σ] = 1:n_chains
    end
	return xs, norm_const, r, index_process, inds
end
