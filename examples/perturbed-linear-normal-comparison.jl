using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

# Number of perturbation terms; c = 4·softmax(t) so Σcᵢ = 4 automatically
const N_TERMS = 6

function make_config()
    optimizer_config = Dict([
        "DoWG"   => nothing,
        "DoG"   => nothing,
        "no_opt" => nothing,
    ])

    optimizer_colors = Dict(
        "DoWG"   => colorant"#d62728",  # red  — PerturbedLinearPath
        "DoG"   => colorant"#35912f",  # red  — PerturbedLinearPath
        "no_opt" => colorant"#9e9e9e",  # gray — LinearPath baseline
    )

    pt_config = NamedTuple([
        :steps_per_round => 100,
        :n_rounds        => 40,
        :n_chains        => 50,
    ])

    return NamedTuple([
        :pt_config         => pt_config,
        :optimizer_config  => optimizer_config,
        :optimizer_colors  => optimizer_colors,
    ])
end

make_init_schedule(n_chains) = collect(range(0, 1, n_chains))
make_x0(n_chains) = zeros(n_chains)

# Normal reference N(0,1) → Normal target N(5,1): a mild but non-trivial bridge
function make_sampling_problem()
    D0 = Normal(0.0, 1.0)
    D1 = Normal(100.0, 1.0)
    return NormalProblem(D0, D1)
end

# PerturbedLinearPath for optimized runs, LinearPath as no-opt baseline
function make_path(optimized::Bool)
    if optimized
        return PerturbedLinearPath(N_TERMS, AutoForwardDiff())
    else
        return LinearPath()
    end
end

function make_path_problem(problem, path)
    PathProblem(problem, path, IterExplorer(SliceSampler(), 5))
end

# Constraint is baked into the softmax reparametrisation; no projection needed
function make_prox_opt()
    return ProjectionState(SumConstraint(1.))
end

function make_optimizer(opt_state)
    return (typeof(opt_state) != NoOptState
        ? ProximalStochOptState(opt_state, make_prox_opt())
        : opt_state)
end

function run_problem!(runs, name, opt_state, pt_config)
    optimized = typeof(opt_state) != NoOptState
    problem   = make_sampling_problem()
    path      = make_path(optimized)
    ptproblem = make_path_problem(problem, path)
    optimizer = make_optimizer(opt_state)

    n_chains       = pt_config.n_chains
    x0             = make_x0(n_chains)
    init_schedule  = make_init_schedule(n_chains)

    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=pt_config.n_rounds,
        steps_per_round=n -> pt_config.steps_per_round,
        objective=SKLObjective()
    )
end

function color_per_opt(optimizers, optimizer_colors)
    colors = Dict{String, Any}()
    for (name, _) in optimizers
        # Match color by suffix (strip "SKL-" prefix added by make_optimizers)
        for (key, c) in optimizer_colors
            if endswith(name, key)
                colors[name] = c
                break
            end
        end
        if !haskey(colors, name)
            colors[name] = colorant"#000000"
        end
    end
    return colors
end

directory(config) = "perturbed-linear-normal"

function run(config; runs=Dict(), run_fn="run.jld2", config_fn="config.jld2")
    optimizers, colors = setup_optimizers(config)
    for (name, (_, opt_state)) in optimizers
        if !(name in keys(runs))
            @info "Running NRPT with $name"
            run_problem!(runs, name, opt_state, config.pt_config)
        else
            @info "Found run with $name. Skipping..."
        end
    end
    mkpath(directory(config))
    jldsave(joinpath(directory(config), config_fn); config)
    filtered = filter_runs(runs)
    jldsave(joinpath(directory(config), run_fn); filtered)
    return runs, colors
end

function filter_runs(runs)
    filtered = Dict{String, Any}()
    for (name, run) in runs
        filtered[name] = NamedTuple([
            :x              => run.x,
            :schedules      => run.schedules,
            :Λ_rej          => run.Λ_rej,
            :Λ_acc          => run.Λ_acc,
            :objective_vals => run.objective_vals,
            :index_process  => run.index_process,
            :rejections     => run.rejections,
            :logZsf         => run.logZsf,
            :logZsb         => run.logZsb,
        ])
    end
    return filtered
end

function setup_optimizers(config)
    optimizers = make_optimizers(config.optimizer_config)
    colors     = color_per_opt(optimizers, config.optimizer_colors)
    return optimizers, colors
end


function show(config, runs, colors)

    dir = directory(config)
    n_rounds = config.pt_config.n_rounds
    n_chains = config.pt_config.n_chains
    steps    = config.pt_config.steps_per_round

    p = compare_barriers(runs, n_rounds, colors; window_size=1)
    Plots.savefig(p, joinpath(dir, "barriers.svg"))

    p = compare_barriers(runs, n_rounds, colors; window_size=20)
    Plots.savefig(p, joinpath(dir, "barriers-smoothed.svg"))

    p = compare_params(runs, colors)
    Plots.savefig(p, joinpath(dir, "param-evolution.svg"))

    n = 10
    p = compare_rt_barrier_final_n(runs, n, steps)
    Plots.savefig(p, joinpath(dir, "barrier-rt-last-$n.svg"))

    p = compare_rt_barrier_cumulative(runs)
    Plots.savefig(p, joinpath(dir, "barrier-rt-cumulative.svg"))

    for (name, _) in runs
        p = plot_density(runs, name, steps, n_chains)
        Plots.savefig(p, joinpath(dir, "density-$name.svg"))
    end

    return runs, colors
end
