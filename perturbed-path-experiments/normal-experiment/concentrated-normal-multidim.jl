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

const DIM = 5
const MU = 2.0
const SIGMA = 0.1
const N_CHAINS = 50 
const N_ROUNDS = 20 
const ORDERS = [1] 

function make_colors()
    devons = collect(palette(:devon, 5))[end-1:-1:1]
    return Dict(
        "linear" => :black,
        "perturbed-1" => devons[4],
    )
end

function make_name_map()
    return Dict(
        "linear" => "Linear",
        "perturbed-1" => "Perturbed linear",
    )
end
function make_name_map2()
    return Dict(
        "linear" => "linear path",
        "perturbed-1" => "perturbed linear path",
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

function make_config(problem, order::Union{Int, Nothing})
    path = make_path(order)
    ptproblem = PathProblem(problem, path, SliceSampler())
    optimizer = make_optimizer(order)
    return NRPTConfig(
        ptproblem, optimizer;
        n_chains = N_CHAINS,
        n_rounds = N_ROUNDS,
        steps_per_round = n -> 2^n,
        threaded = true,
        record_samples = false,
        progress = true
    )
end

directory() = joinpath(@__DIR__, "normal-results-0.1-$DIM-dimensional-long")

function run_all(; runs = Dict())
    problem = make_sampling_problem()

    if !("linear" in keys(runs))
        @info "Running LinearPath baseline"
        runs["linear"] = optimized_nrpt(make_config(problem, 0))
    else
        @info "Found linear, skipping..."
    end

    for order in ORDERS
        name = "perturbed-$order"
        if !(name in keys(runs))
            @info "Running $name"
            runs[name] = optimized_nrpt(make_config(problem, order))
        else
            @info "Found $name, skipping..."
        end
    end

    mkpath(directory())
    return runs
end

function show_results(runs)
    dir = directory()
    mkpath(dir)
    colors = make_colors()
    name_map = make_name_map()
    name_map2 = make_name_map2()
    title = "Normal interpolation, 1d." 

    p = compare_barriers(title, runs, colors, name_map)
    println(p)
    save(joinpath(dir, "barriers.pdf"), p)

    p = compare_params(title, Dict("perturbed-1" => runs["perturbed-1"]), colors, name_map)
    save(joinpath(dir, "param-evolution.pdf"), p)

    Λ_linear = runs["linear"].schedule_recorder.Λ_rej[end]
    E_linear = runs["linear"].schedule_recorder.Λ_acc[end]
    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=Λ_linear, ineff=E_linear)
    save(joinpath(dir, "cumulative-round-trips-iter-ineff-asympt.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=Λ_linear, ineff=nothing)
    save(joinpath(dir, "cumulative-round-trips-iter-asympt.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=nothing, ineff=E_linear)
    save(joinpath(dir, "cumulative-round-trips-iter-ineff.pdf"), p)

 

    # p = compare_params(runs, colors, name_map)
    # save(joinpath(dir, "param-evolution.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map)
    save(joinpath(dir, "cumulative-round-trips-iter.pdf"), p)

    for name in keys(runs)
        p = plot_density(title, runs, name, N_CHAINS, name_map2, "x")
        save(joinpath(dir, "last_round_density-$name.pdf"), p)
    end

    return 
end