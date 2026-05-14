mutable struct IndexProcess
    proc::Matrix{Int}
    rounds::Vector{Int}
    iteration::Int
end

function IndexProcess(n_chains::Int, rounds::Vector{Int}, initial_index)
    proc = Matrix{Int}(undef, n_chains, sum(rounds))
    proc[:, 1] = initial_index
    return IndexProcess(proc, rounds, 2)
end

function record!(ind_proc::IndexProcess, curr_proc)
    ind_proc.proc[:, ind_proc.iteration] = curr_proc
    return ind_proc.iteration += 1
end

function get_round_index_proc(ind_proc::IndexProcess, round)
    if round == 0
        return @view(ind_proc.proc[:, 1:1])
    else
        rounds = ind_proc.rounds
        s = sum(rounds[1:round])
        e = s + rounds[round + 1]
        return @view(ind_proc.proc[:, (s + 1):e])
    end
end

round_trip_completion_iters(ind_proc::IndexProcess) =
    round_trip_completion_iters(ind_proc.proc)

# IndexProcess dispatch — uses exact round boundaries from ind_proc.rounds.

count_chain_round_trips(ind_proc::IndexProcess, chain) =
    count_chain_round_trips(ind_proc.proc, chain)

round_trip_rate(ind_proc::IndexProcess) =
    round_trip_rate(ind_proc.proc)

round_trip_rate(ind_proc::IndexProcess, round) =
    round_trip_rate(get_round_index_proc(ind_proc, round))

function count_round_trips_per_round(ind_proc::IndexProcess)
    n_rounds = length(ind_proc.rounds)
    boundaries = cumsum(ind_proc.rounds)
    rts = zeros(n_rounds)
    n_chains = size(ind_proc.proc, 1)
    for i in 1:n_chains
        for (_, e, _) in count_chain_round_trips(ind_proc, i)[2]
            r = searchsortedfirst(boundaries, e)
            r <= n_rounds && (rts[r] += 1)
        end
    end
    return rts
end
