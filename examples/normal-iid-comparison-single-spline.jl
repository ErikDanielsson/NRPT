using Distributions, StatsPlots, DifferentiationInterface, Mooncake, NRPT, Random, ColorSchemes
using ForwardDiff
using Printf, LogExpFunctions

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

optimizer_config = Dict([
    "DoG" => nothing,
    "DoWG" => nothing,
    "no_opt" => nothing,
    "Adam" => [2., 20.]
    # "AdaGrad" => [2, 2, 0.2, 0.02, 0.002],
])
optimizers = make_optimizers(optimizer_config) 

base_colors = Dict(
    "Adam"    => colorant"#1f77b4",  # blue
    "AdaGrad" => colorant"#ff7f0e",  # orange
    "DoG"     => colorant"#2ca02c",  # green
    "DoWG"    => colorant"#d62728",  # red
    "no-opt"  => colorant"#7f7f7f"   # gray
)

colors = color_per_opt(optimizers, base_colors)

problem_config = NamedTuple([
    :n_knots => 10
])

pt_config = NamedTuple([
    :steps_per_round => n -> 100
    :n_rounds => 100
    :n_chains => 50
    :init_schedule => n_chains -> collect(range(0, 1, n_chains))
    :x0 => n_chains -> ones(n_chains)
])


function make_sampling_problem()
    return NormalProblem(-1., 0.01, 1., 0.01)
end

function make_path(problem, n_knots)
   return SingleSplinePath(n_knots, 1., problem, AutoForwardDiff())
end

function make_path_problem(path)
    PathProblem(path, NormalIIDExplorer())
end


function make_prox_opt(n_knots)
    return ProjectionState(Box(-100ones(n_knots), 100ones(n_knots)))
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
        end
    end
    return runs
end

compare_barriers(runs; window_size=10) = compare_barriers(runs, pt_config.n_rounds; window_size=window_size, colors=colors)
_plot_density(runs, name, inds; kwargs...) = plot_density(runs, name, inds, pt_config.n_chains; kwargs...)