function count_chain_round_trips(ind_proc::Matrix{Int}, chain)
	n_chains, iterations = size(ind_proc)
	round_trips = 0
	inds = Vector{Tuple{Int, Int, Int}}()
	found_end = false
	found_start = false
	start_ind = -1
	end_ind = -1
	for j in 1:iterations
		if ind_proc[chain, j] == 1 
			if found_end && found_start
				round_trips += 1
				push!(inds, (start_ind, end_ind, j))
			end
			start_ind = j
			found_start = true
			found_end = false
		elseif ind_proc[chain, j] == n_chains
			found_end = true 
			end_ind = j
		end
	end	
	return round_trips, inds
end

function round_trip_rate(ind_proc::Matrix{Int})
	n_chains, iterations = size(ind_proc)
	return sum(count_chain_round_trips(ind_proc, i)[1] for i in 1:n_chains) / iterations
end

function count_round_trips_per_round(ind_proc::Matrix{Int}, n_rounds)
	iters_per_round = div(size(ind_proc, 2), n_rounds)
	rts = zeros(n_rounds)
	for i in 1:size(ind_proc, 1)
		ends = count_chain_round_trips(ind_proc, i)[2]
		for (_, e, _) in ends
			r = div(e, iters_per_round)
			rts[r+1] += 1
		end
	end
	return rts
end