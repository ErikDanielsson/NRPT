struct Indices
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

# Perform one round of the deterministic even odds scheme
function DEO(
	x::Vector{T},
	inds::Indices,
	iterations::Int,
	schedule::Vector{Float64},
	problem::PathProblem
) where T
    n_chains = length(schedule)
	# Accumulators
    xs = Matrix{T}(undef, n_chains, iterations)
    r = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_forward = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_backward = Matrix{Float64}(undef, n_chains - 1, iterations)
	index_process = Matrix{Int}(undef, n_chains, iterations)
	norm_const = Vector{T}(undef, iterations)
    for n in 1:iterations
		# Exploration
        Threads.@threads for i in 1:n_chains
			x[inds.σ[i]] = step(problem, x[inds.σ[i]], schedule[i])
        end
		# Communication
        for i in 1:(n_chains - 1)
			lp_swap1 = log_potential(problem.path, x[inds.σ[i]], schedule[i+1])
			lp_swap2 = log_potential(problem.path, x[inds.σ[i+1]], schedule[i])
			lp_reference1 = log_potential(problem.path, x[inds.σ[i]], schedule[i])
			lp_reference2 = log_potential(problem.path, x[inds.σ[i+1]], schedule[i+1])
            α = exp(min(0, 
                (lp_swap1
				+ lp_swap2
				- lp_reference1
				- lp_reference2)
            ))
            r[i, n] = (1 - α)
			lps_forward[i, n] = lp_swap1 - lp_reference1
			lps_backward[i, n] = lp_swap2 - lp_reference2
            A = rand(Bernoulli(α))
            if (i % 2 == n % 2) && A
				inds = swap(inds, i, i + 1)
            end
        end
		xs[inds.σ_inv, n] = x
		index_process[inds.σ, n] = 1:n_chains
    end
	return xs, (lps_forward, lps_backward), r, index_process, inds
end
