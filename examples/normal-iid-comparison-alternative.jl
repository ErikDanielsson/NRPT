using Distributions, StatsPlots, DifferentiationInterface, Mooncake, NRPT, Random, ColorSchemes
using ForwardDiff

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

function make_config()
    optimizer_config = Dict([
        "DoG" => nothing,
        "DoWG" => nothing,
        "no_opt" => nothing,
        "Adam" => [2, 0.2, 0.02],
        "AdaGrad" => [2, 0.2, 0.02],
        "SGD" => [0.2, 0.02]
    ])

    optimizer_colors = Dict(
        "Adam"          => colorant"#1f77b4",  # strong blue
        "AdaGrad"       => colorant"#ff7f0e",  # orange
        "SGD"           => colorant"#6A3D9A", #colorant"#009E73", #colorant"#e6550d",  # darker burnt orange (related but distinct)
        "DoG"           => colorant"#2ca02c",  # green
        "DoWG"          => colorant"#d62728",  # red
        "no_opt"        => colorant"#9e9e9e"   # lighter neutral gray
    )


    problem_config = NamedTuple([
        :n_knots => 5
    ])

    pt_config = NamedTuple([
        :steps_per_round => 300
        :n_rounds => 150
        :n_chains => 50
    ])

    config = NamedTuple([
        :problem_config => problem_config
        :pt_config =>  pt_config
        :optimizer_config =>  optimizer_config
        :optimizer_colors =>  optimizer_colors
    ])
    return config
end

make_init_schedule(n_chains) = collect(range(0, 1, n_chains))
make_x0(n_chains) = ones(n_chains)

function make_sampling_problem()
    return NormalProblem(-1., 0.01, 1., 0.01)
end

function make_path(problem, n_knots)
   return SplinePath(n_knots, 1., problem, AutoMooncake())
end

function make_path_problem(path)
    PathProblem(path, NormalIIDExplorer())
end


function make_prox_opt(n_knots)
    return ProjectionState(Box(-10000ones(2n_knots), 10000ones(2n_knots)))
end

function make_optimizer(optimizer, n_knots)
    return (typeof(optimizer) != NoOptState
        ? ProximalStochOptState(optimizer, make_prox_opt(n_knots))
        : optimizer)
end

function make_pt_problem(n_knots)
    sprob = make_sampling_problem()
    path = make_path(sprob, n_knots)
    pprob = make_path_problem(path)
    return pprob
end


function run_problem!(runs, name, opt_state, pt_config, problem_config)
    # Run the same problem with NRPT
    ptproblem = make_pt_problem(problem_config.n_knots)
    optimizer = make_optimizer(opt_state, problem_config.n_knots)

    n_chains = pt_config.n_chains
    x0 = make_x0(n_chains)
    init_schedule = make_init_schedule(n_chains)
    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=pt_config.n_rounds,
        steps_per_round=n -> pt_config.steps_per_round,
    )
end

directory = "alternative-normal-comparison-knots:$(problem_config.n_knots)"

function setup_optimizers(total_config)
    optimizers = make_optimizers(total_config.optimizer_config) 
    colors = color_per_opt(optimizers, total_config.optimizer_colors)
    return optimizers, colors
end

function run(config; runs=Dict(), run_fn="run.jld2", config_fn="config.jld2")
    optimizers, colors = setup_optimizers(config)
    for (name, optimizer) in optimizers
        if !(name in keys(runs))
            @info "Running NRPT with $name optimizer"
            run_problem!(runs, name, optimizer, config.pt_config, config.problem_config)
        else
            @info "Found NRPT run with $name optimizer. Skipping..."
        end
    end
    jldsave(joinpath(directory, config_fn); config)
    filtered_runs = filter_runs(runs)
    jldsave(joinpath(directory, run_fn); filtered_runs)
    return runs, colors
end

function filter_runs(runs)
    filtered = Dict{String, Any}()
    for (name, run) in runs
        # Filter out everything that is not an array
        content = NamedTuple([
            :x => run.x,
            :schedules => run.schedules,
            :Λ_rej => run.Λ_rej,
            :Λ_acc => run.Λ_acc,
            :SKL_ests => run.SKL_ests,
            :index_process => run.index_process,
            :rejections => run.rejections,
            :logZsf => run.logZsf,
            :logZsb => run.logZsb,
        ])
        filtered[name] = content
    end
    return filtered
end

function plot_all(runs::Dict, config; subsample=1) 
    return plot_all(
        runs,
        config.pt_config.n_rounds,
        config.pt_config.n_chains,
        colors,
        config.pt_config.steps_per_round,
        directory;
        window_size=10,
        subsample_chains=subsample;
        loaded=false
    )
end

function plot_all(runs::Dict, config; subsample=1) 
    return plot_all(
        runs,
        config.pt_config.n_rounds,
        config.pt_config.n_chains,
        colors,
        config.pt_config.steps_per_round,
        directory;
        window_size=10,
        subsample_chains=subsample;
        loaded=false
    )
end