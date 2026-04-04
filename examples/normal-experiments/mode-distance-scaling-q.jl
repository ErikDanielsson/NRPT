using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2, LinearAlgebra

include("../helpers/select-colors.jl")
include("../helpers/create-optimizer-dict.jl")
include("../helpers/plotting.jl")

# 1D normal bridge: reference N(0,1) → target N(mu, 1).
# As mu grows, more chains are needed; n_chains scales linearly with mu.

const DIMENSION       = 1
const N_TERMS         = 1
const SIGMA           = 1.0   # target std dev
const CHAINS_PER_UNIT = 2     # n_chains = round(Int, CHAINS_PER_UNIT * mu), min 10

# Mode distances to sweep over
const MU_VALUES = [20, 30]

function n_chains_for_distance(mu)
    return max(2, round(Int, CHAINS_PER_UNIT * mu))
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

function make_config(mu)
    n_chains = n_chains_for_distance(mu)
    pt_config = (
        steps_per_round = n -> 2^n,
        n_rounds        = 10,
        n_chains        = n_chains,
        max_tr_steps    = 1,
    )
    return (
        mu               = mu,
        pt_config        = pt_config,
        optimizer_config = make_optimizer_config(),
        optimizer_colors = make_optimizer_colors(),
    )
end

make_init_schedule(n_chains) = collect(range(0, 1, n_chains))
make_x0(n_chains) = [zeros(DIMENSION) for _ in 1:n_chains]

function make_sampling_problem(mu)
    D0 = MvNormal(zeros(DIMENSION), 1.0I)
    D1 = MvNormal(mu * ones(DIMENSION), SIGMA^2 * I)
    return GenericDistributionProblem(D0, D1)
end

function make_path(optimized::Bool)
    if optimized
        return PPathQ(10, AutoForwardDiff())
    else
        return LinearPath()
    end
end

function make_path_problem(problem, path)
    PathProblem(problem, path, IterExplorer(SliceSampler(), 5))
end

function make_optimizer(opt_state)
    if typeof(opt_state) == NoOptState || typeof(opt_state) <: NewtonTrustRegionState
        return opt_state
    else
        return ProximalStochOptState(opt_state, NoProx())
    end
end

function run_problem!(runs, name, opt_state, config)
    pt_config = config.pt_config
    mu        = config.mu
    optimized = typeof(opt_state) != NoOptState
    is_newton = typeof(opt_state) <: NewtonTrustRegionState
    k         = pt_config.max_tr_steps

    problem   = make_sampling_problem(mu)
    path      = make_path(optimized)
    ptproblem = make_path_problem(problem, path)

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

directory(mu) = joinpath("normal-mode-distance-q", "mu-$mu")

function run_distance(mu; runs=Dict(), run_fn="run.jld2", config_fn="config.jld2")
    config    = make_config(mu)
    optimizers = make_optimizers(config.optimizer_config)

    for (name, (_, opt_state)) in optimizers
        if !(name in keys(runs))
            @info "mu=$mu, n_chains=$(config.pt_config.n_chains): Running NRPT with $name"
            run_problem!(runs, name, opt_state, config)
        else
            @info "mu=$mu: Found run with $name. Skipping..."
        end
    end

    dir = directory(mu)
    mkpath(dir)
    jldsave(joinpath(dir, config_fn); config)
    filtered = filter_runs(runs)
    jldsave(joinpath(dir, run_fn); filtered)
    return runs, config
end

function run_and_show_all(mu_values=MU_VALUES)
    all_runs = Dict()
    all_configs = Dict()
    for mu in mu_values
        runs, config = run_distance(mu)
        all_runs[mu]    = runs
        all_configs[mu] = config
        show_distance(mu, all_runs[mu], all_configs[mu])
    end
    return all_runs, all_configs
end

function show_distance(mu, runs, config)
    dir      = directory(mu)
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

function show_sweep(all_runs, all_configs, mu_values=MU_VALUES)
    for mu in mu_values
        show_distance(mu, all_runs[mu], all_configs[mu])
    end
end

# Example usage:
#
#   all_runs, all_configs = run_sweep()
#   show_sweep(all_runs, all_configs)
#
# Or for a single distance:
#
#   runs, config = run_distance(10)
#   show_distance(10, runs, config)
