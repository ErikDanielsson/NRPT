using Distributions, StatsPlots, NRPT, Random, Colors
using ForwardDiff, JLD2, LinearAlgebra, DifferentiationInterface

include("../../examples/helpers/select-colors.jl")
include("../../examples/helpers/create-optimizer-dict.jl")
include("../../examples/helpers/plotting.jl")

# Anneal between two concentrated, separated normal distributions:
#   reference  D0 = N(0,       SIGMA²·I)
#   target     D1 = N(MU·1,    SIGMA²·I)
# Both are concentrated (small SIGMA) and separated (large MU).
# Compare LinearPath baseline vs ScalingGBMPath of various Bernstein orders.

const DIM = 1 
const MU = 2.0
const SIGMA = 0.1
const N_CHAINS = 50 
const N_ROUNDS = 16
const ORDERS = [1] 
const SLICE_SETTINGS = ["large", "small"]
function make_colors()
    return Dict(
        "linear-small"    => colorant"#60A5FA",  # light blue
        "linear-large"    => colorant"#1E3A8A",  # deep blue
        "perturbed-small" => colorant"#FB923C",  # light orange
        "perturbed-large" => colorant"#9A3412",  # deep orange
    )
end
# function make_colors()
#     return Dict(
#         "linear-small" => :gray,
#         "linear-large" => :blue,
#         "perturbed-small" => :red,
#         "perturbed-large" => :green,
#     )
# end

function make_name_map()
    return Dict(
        "linear-small" => "Linear, small step",
        "linear-large" => "Linear, large step",
        "perturbed-small" => "Perturbed, small step",
        "perturbed-large" => "Perturbed, large step",
    )
end
function make_name_map2()
    return Dict(
        "linear-small" => "Linear, small step",
        "linear-large" => "Linear, large step",
        "perturbed-small" => "Perturbed, small step",
        "perturbed-large" => "Perturbed, large step",
    )
end

function make_path(order::Union{Int, Nothing})
    paths = Dict(
       0 => LinearPath,
       1 => SymmetricPerturbed,
    )
    return paths[order]()
end
function make_sampling_problem()
    D0 = MvNormal(zeros(DIM), SIGMA^2 * I)
    D1 = MvNormal(MU * ones(DIM), SIGMA^2 * I)
    return GenericDistributionProblem(D0, D1)
end

function make_optimizer(order::Union{Int, Nothing})
    return order == 0 ? NoOptState() : NewtonTrustRegionState(AutoForwardDiff(), DecayrESSCriterion(10, 0.5))
end

function make_config(problem, order::Union{Int, Nothing}, slice_setting)
    if slice_setting == "large" 
        explorer = SliceSampler(10, 3)
    else
        explorer= SliceSampler(0.1, 1)
    end
    path = make_path(order)
    ptproblem = PathProblem(problem, path, explorer)
    optimizer = make_optimizer(order)
    return NRPTConfig(
        ptproblem, optimizer;
        n_chains = N_CHAINS,
        n_rounds = N_ROUNDS,
        steps_per_round = n -> 2^n,
        threaded = true,
        record_samples = true,
        progress = true
    )
end

directory() = joinpath(@__DIR__, "normal-results-0.1-single-dimensional-step-comp-2")

function run_all(; runs = Dict())
    problem = make_sampling_problem()

    for path in ["linear", "perturbed"]
        for setting in SLICE_SETTINGS
            name = "$path-$setting"
            if !(name in keys(runs))
                @info "Running $name"
                runs[name] = optimized_nrpt(make_config(problem, path == "linear" ? 0 : 1, setting))
            else
                @info "Found $name, skipping..."
            end
        end
    end

    mkpath(directory())
    return runs
end

function show_results(runs; dir=directory())
    mkpath(dir)
    colors = make_colors()
    name_map = make_name_map()
    name_map2 = make_name_map2()
    title = "Gaussian interpolation"

    p = compare_barriers(title, runs, colors, name_map; sch_tick = 2, opt_tick = 1, all_tick = 2)
    println(p)
    save(joinpath(dir, "barriers.pdf"), p)

    p = compare_cumulative_barriers(title, runs, colors, name_map)
    save(joinpath(dir, "cumulative-barriers.pdf"), p)

    p = compare_params(title, Dict(
        "perturbed-large" => runs["perturbed-large"],
        "perturbed-small" => runs["perturbed-small"]
    ), colors, name_map)
    save(joinpath(dir, "param-evolution.pdf"), p)

    Λ_linear = runs["linear-large"].schedule_recorder.Λ_rej[end]
    E_linear = runs["linear-large"].schedule_recorder.Λ_acc[end]
    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=Λ_linear, ineff=E_linear)
    save(joinpath(dir, "cumulative-round-trips-iter-ineff-asympt.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=Λ_linear, ineff=nothing)
    save(joinpath(dir, "cumulative-round-trips-iter-asympt.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=nothing, ineff=E_linear)
    save(joinpath(dir, "cumulative-round-trips-iter-ineff.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map)
    save(joinpath(dir, "cumulative-round-trips-iter.pdf"), p)

    for name in keys(runs)
        p = plot_density(title, runs, name, N_CHAINS, name_map2, "x")
        save(joinpath(dir, "last_round_density-$name.pdf"), p)
        # p = whisker_plot(title, runs, name, N_CHAINS, name_map2, "x")
        # save(joinpath(dir, "last_round_whiskers-$name.pdf"), p)
    end

    return 
end