using Distributions, StatsPlots, DifferentiationInterface, Mooncake, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

function make_config()
    optimizer_config = Dict([
        # "DoG" => nothing,
        "DoWG" => nothing,
        "no_opt" => nothing,
        # "Adam" => [2, 0.2, 0.02],
        # "AdaGrad" => [2, 0.2, 0.02],
        # "SGD" => [0.2, 0.02]
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
        :n_knots => 2
    ])

    pt_config = NamedTuple([
        :steps_per_round => 100
        :n_rounds => 1000
        :n_chains => 10
    ])

    config = NamedTuple([
        :problem_config => problem_config
        :pt_config =>  pt_config
        :optimizer_config => optimizer_config
        :optimizer_colors => optimizer_colors
    ])
    return config
end

make_init_schedule(n_chains) = collect(range(0, 1, n_chains))
make_x0(n_chains) = collect(1.0:n_chains) / (n_chains + 1)

function make_sampling_problem()
    D0 = Normal(0, 10)
    D1 = Beta(3, 3)
    # D1 = Normal(5.0, 1)
    problem = GenericDistributionProblem(D0, D1)
    println(problem)
    return problem
end

function make_path(linear)
    if linear 
        return Path(0.5, AutoForwardDiff())
    else
        return QPath(0.5, AutoForwardDiff())
    end
end

function make_path_problem(problem, path)
    PathProblem(problem, path, IterExplorer(SliceSampler(), 5))
end


function make_prox_opt(n_knots)
    return ProjectionState(Box(-10., 10.))
end

function make_optimizer(optimizer, n_knots)
    return (typeof(optimizer) != NoOptState
        ? ProximalStochOptState(optimizer, make_prox_opt(n_knots))
        : optimizer)
end

function make_pt_problem(linear)
    sprob = make_sampling_problem()
    path = make_path(linear)
    pprob = make_path_problem(sprob, path)
    return pprob
end


function run_problem!(runs, name, opt_state, pt_config, problem_config)
    # Run the same problem with NRPT
    ptproblem = make_pt_problem(typeof(opt_state) == NoOptState)
    optimizer = make_optimizer(opt_state, problem_config.n_knots)

    n_chains = pt_config.n_chains
    x0 = make_x0(n_chains)
    init_schedule = make_init_schedule(n_chains)
    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=pt_config.n_rounds,
        steps_per_round=n -> pt_config.steps_per_round,
        objective=BarrierObjective()
    )
end


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
    mkpath(directory(config))
    jldsave(joinpath(directory(config), config_fn); config)
    filtered_runs = filter_runs(runs)
    jldsave(joinpath(directory(config), run_fn); filtered_runs)
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
            :objective_vals => run.objective_vals,
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
        directory(config);
        window_size=10,
        subsample_chains=subsample,
        loaded=false
    )
end
directory(config) = "normal-qpath"

function plot_all(runs::Dict, config; subsample=1) 
    return plot_all(
        runs,
        config.pt_config.n_rounds,
        config.pt_config.n_chains,
        colors,
        config.pt_config.steps_per_round,
        directory(config);
        window_size=10,
        subsample_chains=subsample,
        loaded=false,
        show_exponents=false
    )
end