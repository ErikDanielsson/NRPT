using Distributions, StatsPlots, DifferentiationInterface, Mooncake, NRPT, Random, ColorSchemes
using ForwardDiff

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

optimizer_config = Dict([
    "DoG" => nothing,
    "DoWG" => nothing,
    "no_opt" => nothing,
    "Adam" => [2, 0.2, 0.02],
    "AdaGrad" => [2, 0.2, 0.02],
    "ScaledAdaGrad" => [0.02, 0.2, 2.0],
])
optimizers = make_optimizers(optimizer_config) 

base_colors = Dict(
    "Adam"          => colorant"#1f77b4",  # strong blue
    "AdaGrad"       => colorant"#ff7f0e",  # orange
    "ScaledAdaGrad" => colorant"#6A3D9A", #colorant"#009E73", #colorant"#e6550d",  # darker burnt orange (related but distinct)
    "DoG"           => colorant"#2ca02c",  # green
    "DoWG"          => colorant"#d62728",  # red
    "no_opt"        => colorant"#9e9e9e"   # lighter neutral gray
)

colors = color_per_opt(optimizers, base_colors)

problem_config = NamedTuple([
    :n_knots => 2
])

pt_config = NamedTuple([
    :steps_per_round => n -> 300
    :n_rounds => 150
    :n_chains => 50
    :init_schedule => n_chains -> collect(range(0, 1, n_chains))
    :x0 => n_chains -> ones(n_chains)
])


function make_sampling_problem()
    return NormalProblem(-1., 0.01, 1., 0.01)
end

function make_path(problem, n_knots)
   return PaperSplinePath(n_knots, 1., problem, AutoForwardDiff())
end

function make_path_problem(path)
    PathProblem(path, NormalIIDExplorer())
end


function make_prox_opt(n_knots)
    return ProjectionState(Box(-10ones(2n_knots), 10ones(2n_knots)))
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


function run_problem!(runs, name, opt_state)
    # Run the same problem with NRPT
    ptproblem = make_pt_problem(problem_config.n_knots)
    optimizer = make_optimizer(opt_state, problem_config.n_knots)

    n_chains = pt_config.n_chains
    x0 = pt_config.x0(n_chains)
    init_schedule = pt_config.init_schedule(n_chains)
    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=pt_config.n_rounds,
        steps_per_round=pt_config.steps_per_round,
    )
end

function run(; runs=Dict())
    for (name, optimizer) in optimizers
        if !(name in keys(runs))
            @info "Running NRPT with $name optimizer"
            run_problem!(runs, name, optimizer)
        else
            @info "Found NRPT run with $name optimizer. Skipping..."
        end
    end
    return runs
end

directory = "paper-normal-comparison-knots:$(problem_config.n_knots)"
plot_all(runs; subsample=1) = plot_all(
    runs,
    pt_config.n_rounds,
    pt_config.n_chains,
    colors,
    pt_config.steps_per_round(pt_config.n_rounds),
    directory;
    window_size=10,
    subsample_chains=subsample
)