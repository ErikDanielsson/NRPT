using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2, LinearAlgebra

include("../helpers/select-colors.jl")
include("../helpers/create-optimizer-dict.jl")
include("../helpers/plotting.jl")

# 1D normal bridge: reference N(0,1) → target N(concentration, 1).
# As concentration grows, more chains are needed; n_chains scales linearly with concentration.

const CHAINS_PER_UNIT = 3     # n_chains = round(Int, CHAINS_PER_UNIT * concentration), min 10

# Mode distances to sweep over
const CONCENTRATION = [10]

function n_chains_for_distance(concentration)
    return max(2, round(Int, CHAINS_PER_UNIT * concentration))
end

function make_optimizer_config()
    Dict([
        "Newton" => nothing,
        "no_opt" => nothing,
    ])
end

function make_optimizer_colors()
    Dict(
        "SKL-Newton"  => colorant"#1f77b4",  # blue — Newton trust region
        "SKL-no_opt"  => colorant"#9e9e9e",  # gray — LinearPath baseline
    )
end

function make_config(concentration)
    n_chains = n_chains_for_distance(concentration)
    pt_config = (
        steps_per_round = 1000,
        n_rounds        = 30,
        n_chains        = n_chains,
        max_tr_steps    = 1,
    )
    return (
        concentration    = concentration,
        pt_config        = pt_config,
        optimizer_config = make_optimizer_config(),
        optimizer_colors = make_optimizer_colors(),
    )
end

make_init_schedule(n_chains) = collect(range(0, 1, n_chains))
make_x0(n_chains) = [0.5ones(2) for _ in 1:n_chains]

function make_path(optimized::Bool)
    if optimized
        return PPathQ(3, AutoForwardDiff())
    else
        return LinearPath()
    end
end

function make_path_problem(concentration, path)
    concentration = 10^concentration
    problem, explorer = NRPT.unidentifiable_product_slice_sampler(10 * concentration, 5 * concentration)
    # problem, explorer = NRPT.mvnormal_slice_sampler(2; mu=10)
    PathProblem(problem, path, explorer)
end

function make_optimizer(opt_state)
    if typeof(opt_state) == NoOptState || typeof(opt_state) <: NewtonTrustRegionState
        return opt_state
    else
        return ProximalStochOptState(opt_state, NoProx())
    end
end

function run_problem!(runs, name, opt_state, config)
    pt_config     = config.pt_config
    concentration = config.concentration
    optimized     = typeof(opt_state) != NoOptState
    is_newton     = typeof(opt_state) <: NewtonTrustRegionState
    k             = pt_config.max_tr_steps

    path      = make_path(optimized)
    ptproblem = make_path_problem(concentration, path)

    if is_newton
        optimizer     = opt_state
        n_rounds      = pt_config.n_rounds ÷ k
        steps_per_rnd = pt_config.steps_per_round * k
    else
        optimizer     = make_optimizer(opt_state)
        n_rounds      = pt_config.n_rounds
        steps_per_rnd = pt_config.steps_per_round
    end

    n_chains      = pt_config.n_chains
    x0            = make_x0(n_chains)
    init_schedule = make_init_schedule(n_chains)

    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=_ -> steps_per_rnd,
        objective=SKLObjective()
    )
end

function filter_runs(runs)
    filtered = Dict{String, Any}()
    for (name, run) in runs
        nt = (
            x              = run.x,
            schedules      = run.schedules,
            Λ_rej          = run.Λ_rej,
            Λ_acc          = run.Λ_acc,
            objective_vals = run.objective_vals,
            index_process  = run.index_process,
            rejections     = run.rejections,
            logZsf         = run.logZsf,
            logZsb         = run.logZsb,
        )
        if hasproperty(run, :opt_state) && hasproperty(run.opt_state, :min_eigvals)
            nt = merge(nt, (; min_eigvals = copy(run.opt_state.min_eigvals)))
        end
        filtered[name] = nt
    end
    return filtered
end

directory(concentration) = joinpath("normal-mode-distance-q", "concentration-$concentration")

function run_distance(concentration; runs=Dict(), run_fn="run.jld2", config_fn="config.jld2")
    config    = make_config(concentration)
    optimizers = make_optimizers(config.optimizer_config)

    for (name, (_, opt_state)) in optimizers
        if !(name in keys(runs))
            @info "concentration=$concentration, n_chains=$(config.pt_config.n_chains): Running NRPT with $name"
            run_problem!(runs, name, opt_state, config)
        else
            @info "concentration=$concentration: Found run with $name. Skipping..."
        end
    end

    dir = directory(concentration)
    mkpath(dir)
    # jldsave(joinpath(dir, config_fn); config)
    # filtered = filter_runs(runs)
    # jldsave(joinpath(dir, run_fn); filtered)
    return runs, config
end

function run_and_show_all(concentrations=CONCENTRATION)
    all_runs = Dict()
    all_configs = Dict()
    for concentration in concentrations
        runs, config = run_distance(concentration)
        all_runs[concentration]    = runs
        all_configs[concentration] = config
        # show_distance(concentration, all_runs[concentration], all_configs[concentration])
    end
    return all_runs, all_configs
end

function show_distance(concentration, runs, config)
    dir      = directory(concentration)
    mkpath(dir) 
    n_rounds = config.pt_config.n_rounds
    steps    = config.pt_config.steps_per_round
    n_chains = config.pt_config.n_chains
    colors   = make_optimizer_colors()

    p = compare_barriers(runs, n_rounds, colors; window_size=1)
    Plots.savefig(p, joinpath(dir, "barriers.svg"))

    p = compare_barriers(runs, n_rounds, colors; window_size=20)
    Plots.savefig(p, joinpath(dir, "barriers-smoothed.svg"))

    p = compare_cumulative_barriers(runs, colors)
    Plots.savefig(p, joinpath(dir, "cumulative-barriers.svg"))

    p = compare_params(runs, colors)
    Plots.savefig(p, joinpath(dir, "param-evolution.svg"))

    p = compare_min_eigvals(runs, colors)
    Plots.savefig(p, joinpath(dir, "min-eigvals.svg"))

    n = 10
    p = compare_rt_barrier_final_n(runs, n, steps)
    Plots.savefig(p, joinpath(dir, "barrier-rt-last-$n.svg"))

    p = compare_rt_barrier_cumulative(runs)
    Plots.savefig(p, joinpath(dir, "barrier-rt-cumulative.svg"))

    p = cumulative_round_trips(runs, n_rounds, colors)
    Plots.savefig(p, joinpath(dir, "cumulative-round-trips.svg"))

    density_dir = joinpath(dir, "densities")
    for (name, _) in runs
        p = plot_density(runs, name, 2 * steps, n_chains; dim=1)
        Plots.savefig(p, joinpath(dir, "final-density-$name.svg"))
        n_points = 10
        step = n_rounds ÷ n_points
        n_rounds_to_show = 2
        name_density_dir = joinpath(density_dir, name)
        mkpath(name_density_dir)
        for round in n_rounds_to_show:step:n_rounds-1
            p = plot_density(runs, name, 2 * steps, n_chains; dim=1, final_ind = 1 + round * steps)
            Plots.savefig(p, joinpath(name_density_dir, "density-round-$round-$name.svg"))
        end
    end
end

function density2d_contour(samples::Matrix)
    x = samples[1, :]
    y = samples[2, :]
    k = kde((x, y))
    contourf(k, c=:vik, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
end

function show_sweep(all_runs, all_configs, concentration=CONCENTRATION)
    for concentration in concentration
        show_distance(concentration, all_runs[concentration], all_configs[concentration])
    end
end