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
        x_round, _, r, ind_proc, inds = DEO(x[:, end], inds, iterations, schedule, problem)
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
    problem::PathProblem,
    opt_state::Optimizer,
    seed = 2;
    n_rounds = 10,
    use_accept=false,
    steps_per_round=n -> 100,
    save_rejection=false,
    save_lps=false,
    record_samples=true,
    min_opt_ess=100,
    objective::PathObjective=SKLObjective()
) where {T}

    Random.seed!(seed)
    n_chains = length(schedule)
    all_iterations = [steps_per_round(n) for n in 1:n_rounds for _ in 1:2]
    prepend!(all_iterations, 1)
    schedules = Matrix{Float64}(undef, n_chains, length(all_iterations) ÷ 2 + 1)
    schedules[:, 1] = schedule
    barriers = Vector{Any}(undef, n_rounds)
    Λ_rej = Vector{Float64}(undef, n_rounds)
    Λ_acc = Vector{Float64}(undef, n_rounds)
    logZsf = Vector{Float64}(undef, n_rounds)
    logZsb = Vector{Float64}(undef, n_rounds)
    rejections = Matrix{Float64}(undef, n_chains - 1, 0)
    lp_recorder = LPRecorder(Val(save_lps), all_iterations[2:2:end], n_chains)
    sample_recorder = make_sample_recorder(record_samples, n_chains, all_iterations, x0)

    loss_recorder = 

    opt_state = init(problem, opt_state)

    total_iters = sum(all_iterations)
    # x = Matrix{T}(undef, n_chains, total_iters)
    # x[:, 1] = x0
    index_process = IndexProcess(n_chains, all_iterations, 1:n_chains)
    # index_process[:, 1] = 1:n_chains
    col = 2

    chains = PTChains(x0, schedule)
    progress = Progress(total_iters; desc="Running optimized NRPT...", enabled=true)
    ProgressMeter.update!(progress, 1, force=true, showvalues=[
        ("objective", nothing),
        ("Λ", nothing),
        ("η", nothing),
        ("ϕ", extract_reparam(problem.path))
    ])
    for n in 1:n_rounds
        iterations = all_iterations[2n]

        # Set up the chains
        refresh_chains!(chains, schedule, iterations)
        # Run DEO
        (lpsf, lpsb), r = DEO!(chains, problem, index_process, sample_recorder)
        # Log some info
        new_logZf = stepping_stone(lpsf)
        new_logZb = -stepping_stone(lpsb)
        logZsf[n] = new_logZf
        logZsb[n] = new_logZb
        rejections = save_rejections(rejections, r; save=save_rejection)
        # Adapt the schedule
        Λ_rej[n] = compute_Λ(r, schedule; use_accept=false)
        Λ_acc[n] = compute_Λ(r, schedule; use_accept=true)
        Λ_β, schedule = make_schedule(r, schedule; use_accept=use_accept)
        schedules[:, n + 1] = schedule
        barriers[n] = Λ_β
        col += iterations

        iterations = all_iterations[2n + 1]

        # Set up the for the chains for the next iteration 
        refresh_chains!(chains, schedule, iterations)
        # Run DEO
        (lpsf, lpsb), r = DEO!(chains, problem, index_process, sample_recorder)
        record_lps!(lp_recorder, n, stack([lpsf, lpsb], dims=3))

        # Adapt the path
        if iterations > min_opt_ess
            obj_val = adapt_path!(problem, chains, opt_state, objective)
        else
            obj_val = nothing
        end

        # Save some stuff
        objective_vals[n] = obj_val
        Λ = compute_Λ(r, schedule; use_accept=false)

        col += iterations
        ProgressMeter.update!(
            progress,
            col,
            force=true,
            showvalues=[
                ("objective", obj_val),
                ("Λ", Λ),
                ("η", get_last_eta(opt_state)),
                ("ϕ", extract_reparam(problem.path))
            ]
        )
    end
    ProgressMeter.update!(
        progress,
        total_iters,
        force=true,
    )
    return NamedTuple([
        :x => sample_recorder,
        :schedules => schedules,
        :barriers => barriers,
        :Λ_rej => Λ_rej,
        :Λ_acc => Λ_acc,
        :objective_vals => objective_vals,
        :index_process => index_process,
        :opt_state =>  opt_state,
        :rejections => rejections,
        :logZsf => logZsf,
        :logZsb => logZsb,
        :problem => problem,
        :lp_recorder => lp_recorder
    ])
end

# If we want to run a comparable NRPT without an optimizer
optimized_nrpt(
    x0::Vector{T},
    schedule::Vector{Float64},
    problem::PathProblem,
    seed = 2;
    kwargs...
) where {T} = optimized_nrpt(x0, schedule, problem, NoOptState(), seed; kwargs...)
    
# If we want to run a comparable NRPT without an optimizer
function optimized_nrpt(
    n_chains::Int,
    problem::PathProblem,
    seed = 2;
    kwargs...
)
    schedule = collect(range(0, 1, n_chains))
    x0 = [sample_iid(problem.problem) for _ in 1:n_chains]
    return optimized_nrpt(x0, schedule, problem, NoOptState(), seed; kwargs...)
end
    

