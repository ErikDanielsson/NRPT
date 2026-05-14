# Perform one round of the deterministic even odds scheme
function DEO!(
        chains::PTChains{N, T, Tr},
        problem::PathProblem,
        index_process::IndexProcess,
        sample_recorder::MaybeSampleRecorder{T}
    ) where {N, T, Tr}
    n_chains, iterations = size(chains)
    # Accumulators
    rejections = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_forward = Matrix{Float64}(undef, n_chains - 1, iterations)
    lps_backward = Matrix{Float64}(undef, n_chains - 1, iterations)
    for n in 1:iterations
        # Exploration
        explore!(problem, chains, n)

        # Communication
        r, lp_forward, lp_backward = swap_chains(problem, chains, n)

        # Record swap statistics
        rejections[:, n] = r
        lps_forward[:, n] = lp_forward
        lps_backward[:, n] = lp_backward

        # Record the state and index process
        record!(sample_recorder, chains)
        record!(index_process, get_index_process(chains))
    end
    return (lps_forward, lps_backward), rejections
end
