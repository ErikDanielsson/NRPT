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
	warmup = 1,
    use_accept=false,
    steps_per_round=n -> 100,
    save_rejection=false,
    save_lps=false,
    objective::PathObjective=SKLObjective()
) where {T}

    Random.seed!(seed)
    n_chains = length(schedule)
    all_iterations = [steps_per_round(n) for n = warmup:n_rounds]
    schedules = Matrix{Float64}(undef, n_chains, length(all_iterations) + 1)
    schedules[:, 1] = schedule
    barriers = Vector{Any}(undef, length(all_iterations))
    Λ_rej = Vector{Float64}(undef, length(all_iterations))
    Λ_acc = Vector{Float64}(undef, length(all_iterations))
    objective_vals = Vector{Union{Float64, Nothing}}(undef, length(all_iterations))
    logZsf = Vector{Float64}(undef, length(all_iterations))
    logZsb = Vector{Float64}(undef, length(all_iterations))
    rejections = Matrix{Float64}(undef, n_chains - 1, 0)
    lp_recorder = LPRecorder(Val(save_lps), all_iterations, n_chains)

    loss_averager = init_averager(Float64(n_chains), PolynomialDecayAverager(0.00))
    Λ_averager = init_averager(Float64(n_chains), PolynomialDecayAverager(0.00))

    opt_state = init(problem, opt_state)

    total_iters = sum(all_iterations)
    x = Matrix{T}(undef, n_chains, 1 + 2 * total_iters)
    x[:, 1] = x0
    index_process = Matrix{Int}(undef, n_chains, 1 + 2 * total_iters)
    index_process[:, 1] = 1:n_chains
    col = 2

    chains = PTChains(x0, schedule)
    progress = Progress(length(all_iterations); desc="Running optimized NRPT...")
    ProgressMeter.update!(progress, 0, force=true, showvalues=[
        ("objective", nothing),
        ("Λ", nothing),
        ("η", nothing),
        ("ϕ", extract_reparam(problem.path))
    ])
    for n in eachindex(all_iterations)
        iterations = all_iterations[n]

        # Set up the chains
        refresh_chains!(chains, schedule, iterations)
        # Run DEO
        x_round, (lpsf, lpsb), r, ind_proc, chains = DEO(chains, problem)
        # Log some info
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
        x[:, col:col+iterations-1] = x_round
        index_process[:, col:col+iterations-1] = ind_proc
        col += iterations

        # Set up the chains
        refresh_chains!(chains, schedule, iterations)
        # Run DEO
        x_round, (lpsf, lpsb), r, ind_proc, chains = DEO(chains, problem)

        record_lps!(lp_recorder, )
        # Adapt the path
        obj_val = adapt_path!(problem, chains, schedule, opt_state, objective)
        # Save some stuff
        objective_vals[n] = obj_val
        Λ = compute_Λ(r, schedule; use_accept=false)
        update!(Λ, n, Λ_averager)
        update!(obj_val, n, loss_averager)
        x[:, col:col+iterations-1] = x_round
        index_process[:, col:col+iterations-1] = ind_proc
        col += iterations
        next!(
            progress,
            showvalues=[
                ("objective", obj_val),
                ("Λ", Λ),
                ("η", get_last_eta(opt_state)),
                ("ϕ", extract_reparam(problem.path))
            ]
        )
    end
    return NamedTuple([
        :x => x,
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
        :problem => problem
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
    
