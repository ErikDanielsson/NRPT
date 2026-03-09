# Perform one round of the deterministic even odds scheme
function DEO(
	x::Vector{T},
	starting_inds::Indices,
	iterations::Int,
	schedule::Vector{Float64},
	problem::PathProblem
) where T
    n_chains = length(schedule)
	# Accumulators
    xs = Matrix{T}(undef, n_chains, iterations)
    rejections = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_forward = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_backward = Matrix{Float64}(undef, n_chains - 1, iterations)
	index_process = Matrix{Int}(undef, n_chains, iterations)

	chains = PTChains(starting_inds, schedule, iterations)
    for n in 1:iterations
		# Exploration
		x = explore(problem, chains, x, n)	

		# Communication
		r, lp_forward, lp_backward = swap_chains(problem, chains, n)	

		# Record swap statistics
		rejections[:, n] = r
		lps_forward[:, n] = lp_forward
		lps_backward[:, n] = lp_backward

		# Record the state and index process
		xs[:, n] = get_state_per_temperature(chains, x)
		index_process[:, n] = get_index_process(chains)
    end
	return xs, (lps_forward, lps_backward), rejections, index_process, chains
end