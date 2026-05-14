function build_all_iterations(n_rounds::Int, min_step_for_opt::Int, steps_per_round)::Vector{Int}
    iters = [1]
    for n in 1:n_rounds
        s = steps_per_round(n)
        push!(iters, s)
        s >= min_step_for_opt && push!(iters, s)
    end
    return iters
end

function run_schedule_round!(
        chains::PTChains,
        problem::PathProblem,
        schedule::Vector{Float64},
        iterations::Int,
        index_process::IndexProcess,
        sample_recorder::MaybeSampleRecorder,
        schedule_recorder::ScheduleRecorder,
        logz_recorder::LogZRecorder;
        use_accept::Bool = false,
    )::Tuple{Vector{Float64}, Float64, Matrix{Float64}}
    refresh_chains!(chains, schedule, iterations)
    (lpsf, lpsb), r = DEO!(chains, problem, index_process, sample_recorder)

    Λ_rej = compute_Λ(r, schedule; use_accept = false)
    Λ_acc = compute_Λ(r, schedule; use_accept = true)
    Λ_β, new_schedule = make_schedule(r, schedule; use_accept = use_accept)
    record!(schedule_recorder, new_schedule, Λ_β, Λ_rej, Λ_acc)
    record_schedule!(logz_recorder, stepping_stone(lpsf), -stepping_stone(lpsb))

    return new_schedule, Λ_rej, r
end

function run_opt_round!(
        chains::PTChains,
        problem::PathProblem,
        opt_state::Optimizer,
        objective::PathObjective,
        schedule::Vector{Float64},
        threaded::Bool,
        opt_round_n::Int,
        index_process::IndexProcess,
        sample_recorder::MaybeSampleRecorder,
        loss_recorder::SKLRecorder,
        logz_recorder::LogZRecorder,
        lp_recorder::Union{LPRecorder, Nothing},
        progress::Bool,
    )
    (lpsf, lpsb), r = DEO!(chains, problem, index_process, sample_recorder)

    record_lps!(lp_recorder, opt_round_n, stack([lpsf, lpsb], dims = 3))
    record_opt!(logz_recorder, stepping_stone(lpsf), -stepping_stone(lpsb))

    if !isa(opt_state, NoOptState)
        obj_val = adapt_path!(problem, chains, opt_state, objective, threaded, progress)
        new_schedule = schedule
    else
        # Compute the objective value (this is a no-op with regards to the path)
        obj_val = adapt_path!(problem, chains, opt_state, objective, threaded, progress)
        # Do schedule adaptation with the samples instead
        _, new_schedule = make_schedule(r, schedule; use_accept = false)
    end
    Λ_opt = compute_Λ(r, schedule; use_accept = false)
    record!(loss_recorder, Λ_opt, obj_val)

    return obj_val, Λ_opt, new_schedule
end

function optimized_nrpt(config::NRPTConfig{T, S}) where {T, S}
    Random.seed!(config.seed)
    n_chains_val = n_chains(config)
    schedule = collect(range(0.0, 1.0, n_chains_val))

    schedule_iters = [config.steps_per_round(n) for n in 1:config.n_rounds]
    opt_iters = [
        config.steps_per_round(n) for n in 1:config.n_rounds
            if config.steps_per_round(n) >= config.min_steps_for_opt
    ]
    n_opt_rounds = length(opt_iters)
    all_iterations = build_all_iterations(config.n_rounds, config.min_steps_for_opt, config.steps_per_round)

    sample_recorder = make_sample_recorder(config.record_samples, n_chains_val, all_iterations, config.x0)
    index_process = IndexProcess(n_chains_val, all_iterations, 1:n_chains_val)
    schedule_recorder = ScheduleRecorder(n_chains_val, config.n_rounds, schedule)
    logz_recorder = LogZRecorder(config.n_rounds, n_opt_rounds)
    loss_recorder = SKLRecorder(n_opt_rounds)
    lp_recorder = LPRecorder(Val(config.save_lps), opt_iters, n_chains_val)
    rejections = Matrix{Float64}(undef, n_chains_val - 1, 0)

    opt_state = init(config.problem, config.opt_state)
    chains = PTChains(config.x0, schedule, config.threaded)

    total_iters = sum(all_iterations)
    col = 2
    obj_val = NaN
    Λ_opt = NaN

    beta_cum_iters = 0
    opt_cum_iters = 0
    opt_round_n = 0
    progress = config.progress
    prog_beta = Progress(sum(schedule_iters); desc = "Schedule optimization", enabled = progress)
    prog_opt = Progress(sum(opt_iters); desc = "Path optimization", offset = 2, enabled = progress)
    ProgressMeter.update!(prog_beta, beta_cum_iters, force = true)
    ProgressMeter.update!(
        prog_opt, opt_cum_iters, force = true, showvalues = [
            ("objective", nothing),
            ("Λ", nothing),
            ("η", nothing),
            ("ϕ", extract_reparam(config.problem.path)),
        ]
    )

    for n in 1:config.n_rounds
        schedule, Λ_rej, r = run_schedule_round!(
            chains, config.problem, schedule, schedule_iters[n],
            index_process, sample_recorder,
            schedule_recorder, logz_recorder;
            use_accept = config.use_accept,
        )
        rejections = save_rejections(rejections, r; save = config.save_rejection)
        beta_cum_iters += schedule_iters[n]
        ProgressMeter.update!(
            prog_beta, beta_cum_iters, force = true, showvalues = [("Λ", Λ_rej)]
        )

        if schedule_iters[n] >= config.min_steps_for_opt
            opt_round_n += 1
            iterations = opt_iters[opt_round_n]
            refresh_chains!(chains, schedule, iterations)
            obj_val, Λ_opt, schedule = run_opt_round!(
                chains, config.problem, opt_state, config.objective, schedule,
                config.threaded, opt_round_n, index_process, sample_recorder,
                loss_recorder, logz_recorder, lp_recorder, progress
            )
            opt_cum_iters += opt_iters[opt_round_n]
            ProgressMeter.update!(
                prog_opt, opt_cum_iters, force = true,
                showvalues = [
                    ("objective", isnan(obj_val) ? nothing : obj_val),
                    ("Λ", Λ_opt),
                    ("η", get_last_eta(opt_state)),
                    ("ϕ", extract_reparam(config.problem.path)),
                ]
            )
        end
    end

    return (
        x = sample_recorder,
        index_process = index_process,
        schedule_recorder = schedule_recorder,
        logz_recorder = logz_recorder,
        loss_recorder = loss_recorder,
        opt_state = opt_state,
        rejections = rejections,
        problem = config.problem,
        lp_recorder = lp_recorder,
    )
end

# Standard NRPT without path optimization
function nrpt(
        x0::Vector{T},
        schedule::Vector{Float64},
        problem::PathProblem{P, E},
        seed = 2;
        n_rounds = 10,
        warmup = 1,
        use_accept = false,
        steps_per_round = n -> 100
    ) where {T, P, E}

    Random.seed!(seed)
    n_chains_val = length(schedule)
    round_iters = [steps_per_round(n) for n in warmup:n_rounds]
    all_iterations = [1; round_iters]

    sample_recorder = make_sample_recorder(true, n_chains_val, all_iterations, x0)
    index_process = IndexProcess(n_chains_val, all_iterations, 1:n_chains_val)
    schedule_recorder = ScheduleRecorder(n_chains_val, length(round_iters), schedule)

    chains = PTChains(x0, schedule)
    progress = Progress(sum(all_iterations); desc = "Running NRPT...")

    for iterations in round_iters
        refresh_chains!(chains, schedule, iterations)
        _, r = DEO!(chains, problem, index_process, sample_recorder)

        Λ_rej = compute_Λ(r, schedule; use_accept = false)
        Λ_acc = compute_Λ(r, schedule; use_accept = true)
        Λ_β, schedule = make_schedule(r, schedule; use_accept = use_accept)
        record!(schedule_recorder, schedule, Λ_β, Λ_rej, Λ_acc)

        next!(progress, showvalues = [("Λ", Λ_β(1.0))])
    end

    return (
        x = sample_recorder,
        schedule_recorder = schedule_recorder,
        index_process = index_process,
    )
end
