using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2, LinearAlgebra

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

# Number of perturbation terms; c = 4·softmax(t) so Σcᵢ = 4 automatically
const N_TERMS = 1
const DIMENSION = 5

function make_config()
    optimizer_config = Dict([
        "DoG"    => nothing,
        "Newton" => nothing,
        "no_opt" => nothing,
    ])

    optimizer_colors = Dict(
        "DoG"       => colorant"#35912f",  # green     — gradient, no trust region
        "DoG-TR"    => colorant"#17becf",  # cyan      — first-order trust region
        "Newton"    => colorant"#1f77b4",  # blue      — Newton trust region
        "no_opt"    => colorant"#9e9e9e",  # gray      — LinearPath baseline
    )

    pt_config = NamedTuple([
        :steps_per_round => 100,
        :n_rounds        => 100,
        :n_chains        => 20,
    ])

    return NamedTuple([
        :pt_config         => pt_config,
        :optimizer_config  => optimizer_config,
        :optimizer_colors  => optimizer_colors,
    ])
end

make_init_schedule(n_chains) = collect(range(0, 1, n_chains))
make_x0(n_chains) = [zeros(DIMENSION) for _ in 1:n_chains] 

# Normal reference N(0,1) → Normal target N(5,1): a mild but non-trivial bridge
function make_sampling_problem()
    D0 = MvNormal(zeros(DIMENSION), 1.0I)
    D1 = MvNormal(10.0ones(DIMENSION), 0.1I)
    return GenericDistributionProblem(D0, D1)
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
    return NoProx()
end

function make_optimizer(opt_state)
    if typeof(opt_state) == NoOptState || typeof(opt_state) <: NewtonTrustRegionState
        return opt_state
    else
        return ProximalStochOptState(opt_state, make_prox_opt())
    end
end

function run_problem!(runs, name, opt_state, pt_config; use_trust_region=false)
    optimized = typeof(opt_state) != NoOptState
    problem   = make_sampling_problem()
    path      = make_path(optimized)
    ptproblem = make_path_problem(problem, path)
    optimizer = make_optimizer(opt_state)
    if optimized && use_trust_region
        optimizer = TrustRegionState(optimizer)
    end

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
    for (name, (_, opt_state)) in optimizers
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
        # Also register the -TR variant if this is not no_opt
        if typeof(opt_state) != NoOptState
            tr_name = "$name-TR"
            for (key, c) in optimizer_colors
                if endswith(tr_name, key)
                    colors[tr_name] = c
                    break
                end
            end
            if !haskey(colors, tr_name)
                colors[tr_name] = colorant"#000000"
            end
        end
    end
    return colors
end

directory(config) = "perturbed-linear-normal"

function run(config; runs=Dict(), run_fn="run.jld2", config_fn="config.jld2")
    optimizers, colors = setup_optimizers(config)
    for (name, (_, opt_state)) in optimizers
        is_no_opt = typeof(opt_state) == NoOptState
        is_newton = typeof(opt_state) <: NewtonTrustRegionState
        variants = (is_no_opt || is_newton) ? [(name, false)] : [(name, false), ("$name-TR", true)]
        for (run_name, use_tr) in variants
            if !(run_name in keys(runs))
                @info "Running NRPT with $run_name"
                run_problem!(runs, run_name, opt_state, config.pt_config; use_trust_region=use_tr)
            else
                @info "Found run with $run_name. Skipping..."
            end
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
