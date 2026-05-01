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
    ind_proc.iteration += 1
end