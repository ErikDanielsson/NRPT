# Perform one round of the deterministic even odds scheme
function DEO(
	chains::PTChains{T},
	problem::PathProblem
) where T
    n_chains, iterations = size(chains)
	# Accumulators
    xs = Matrix{T}(undef, n_chains, iterations)
    rejections = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_forward = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_backward = Matrix{Float64}(undef, n_chains - 1, iterations)
	index_process = Matrix{Int}(undef, n_chains, iterations)

	# println([chain.beta for chain in chains.chains])
    for n in 1:iterations
		# Exploration
		explore!(problem, chains, n)	

		# Communication
		r, lp_forward, lp_backward = swap_chains(problem, chains, n)	

		# println("swap x = $(get_state_per_temperature(chains))")

		# Record swap statistics
		rejections[:, n] = r
		lps_forward[:, n] = lp_forward
		lps_backward[:, n] = lp_backward

		# Record the state and index process
		xs[:, n] = get_state_per_temperature(chains)
		index_process[:, n] = get_index_process(chains)
    end
	return xs, (lps_forward, lps_backward), rejections, index_process, chains
end