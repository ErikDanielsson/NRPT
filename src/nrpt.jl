# Standard NRPT without path optimization
function nrpt(
    x0::Vector{T},
    schedule::Vector{Float64},
    problem::PathProblem{P, E},
    seed = 2;
    n_rounds = 10,
	warmup = 1,
    use_accept=false,
    steps_per_round=n -> 100
) where {T, P, E}

    Random.seed!(seed)
    n_chains = length(schedule)
    all_iterations = [steps_per_round(n) for n = warmup:n_rounds]
    schedules = Matrix{Float64}(undef, n_chains, length(all_iterations) + 1)
    schedules[:, 1] = schedule
    barriers = Vector{Any}(undef, length(all_iterations))

    x = Array{T}(undef, n_chains, 1)
    x[:, 1] = x0

    index_process = Matrix{Int}(undef, n_chains, 1)
    index_process[:, 1] = 1:n_chains
    inds = Indices(n_chains)
    progress = Progress(length(all_iterations); desc="Running NRPT...")
    for n = 1:length(all_iterations)
        iterations = all_iterations[n]
        # Run DEO
        x_round, logZ, r, ind_proc, inds = DEO(x[:, end], inds, iterations, schedule, problem)
        # Adapt the schedule
        b, schedule = make_schedule(r, schedule; use_accept=use_accept)
        schedules[:, n + 1] = schedule
        barriers[n] = b
        index_process = hcat(index_process, ind_proc)
        x = hcat(x, x_round)
        next!(progress, showvalues=[("Λ", b(1.0))])
    end
    return x, schedules, barriers, index_process
end


function optimized_nrpt(
    x0::Vector{T},
    schedule::Vector{Float64},
    problem::PathProblem{P, E},
    opt_state::Optimizer,
    seed = 2;
    n_rounds = 10,
	warmup = 1,
    use_accept=false,
    steps_per_round=n -> 100,
    save_rejection=false
) where {T, P, E}

    Random.seed!(seed)
    n_chains = length(schedule)
    all_iterations = [steps_per_round(n) for n = warmup:n_rounds]
    schedules = Matrix{Float64}(undef, n_chains, length(all_iterations) + 1)
    schedules[:, 1] = schedule
    barriers = Vector{Any}(undef, length(all_iterations))
    Λ_rej = Vector{Float64}(undef, length(all_iterations))
    Λ_acc = Vector{Float64}(undef, length(all_iterations))
    SKL_ests = Vector{Union{Float64, Nothing}}(undef, length(all_iterations))
    logZsf = Vector{Float64}(undef, length(all_iterations))
    logZsb = Vector{Float64}(undef, length(all_iterations))
    rejections = Matrix{Float64}(undef, n_chains - 1, 0)

    opt_state = init(problem, opt_state)

    x = Array{T}(undef, n_chains, 1)
    x[:, 1] = x0

    index_process = Matrix{Int}(undef, n_chains, 1)
    index_process[:, 1] = 1:n_chains
    inds = Indices(n_chains)
    logZ = 0.0
    cumulative_iterations = 0
    progress = Progress(length(all_iterations); desc="Running optimized NRPT...")
    for n in 1:length(all_iterations)
        iterations = all_iterations[n]
        # Run DEO
        x_round, (lpsf, lpsb), r, ind_proc, inds = DEO(x[:, end], inds, iterations, schedule, problem)
        new_logZf = stepping_stone(lpsf)
        new_logZb = -stepping_stone(lpsb)
        logZsf[n] = new_logZf
        logZsb[n] = new_logZb
        rejections = save_rejections(rejections, r; save=save_rejection)
        # Adapt the schedule
        Λ_rej[n] = compute_Λ(r, schedule; use_accept=false)
        Λ_acc[n] = compute_Λ(r, schedule; use_accept=true)
        b, schedule = make_schedule(r, schedule; use_accept=use_accept)
        schedules[:, n + 1] = schedule
        barriers[n] = b
        index_process = hcat(index_process, ind_proc)
        x = hcat(x, x_round)

        x_round, _, r, ind_proc, inds = DEO(x[:, end], inds, iterations, schedule, problem)

        # Adapt the path
        SKL_est = adapt_path!(problem, x_round, schedule, opt_state) 
        # Save some stuff
        SKL_ests[n] = SKL_est
        index_process = hcat(index_process, ind_proc)
        x = hcat(x, x_round)
        next!(progress, showvalues=[("SKL", SKL_est), ("Λ", b(1.0))])
    end
    return NamedTuple([
        :x => x,
        :schedules => schedules,
        :barriers => barriers,
        :Λ_rej => Λ_rej,
        :Λ_acc => Λ_acc,
        :SKL_ests => SKL_ests,
        :index_process => index_process,
        :opt_state =>  opt_state,
        :rejections => rejections,
        :logZsf => logZsf,
        :logZsb => logZsb,
    ])
end

# If we want to run a comparable NRPT without an optimizer
optimized_nrpt(
    x0::Vector{T},
    schedule::Vector{Float64},
    problem::PathProblem{P, E},
    seed = 2;
    kwargs...
) where {T, P, E} = optimized_nrpt(x0, schedule, problem, NoOptState(), seed; kwargs...)
    