using Distributions, StatsPlots, NRPT, Random, Colors
using ForwardDiff, JLD2, LinearAlgebra, DifferentiationInterface

include("../helpers/select-colors.jl")
include("../helpers/create-optimizer-dict.jl")
include("../helpers/plotting.jl")
include("normal-model.jl")

const N_CHAINS = 32 
const N_ROUNDS = 20
const ORDERS = [0, 1, 2, 4] 

function make_colors()
    devons = collect(palette(:devon, 5))[end-1:-1:1]
    return Dict(
        "SKL-no_opt" => :gray,
        "SKL-gbm-0" => :black,
        "SKL-gbm-1" => devons[1],
        "SKL-gbm-2" => devons[2],
        "SKL-gbm-4" => devons[3],
    )
end

function make_name_map()
    return Dict(
        "SKL-no_opt" => "Linear (original space)",
        "SKL-gbm-0" => "Linear (transformed space)",
        "SKL-gbm-1" => "Prior param. (k = 1)",
        "SKL-gbm-2" => "Prior param. (k = 2)",
        "SKL-gbm-4" => "Prior param. (k = 4)",
    )
end

function transform(xs)
    return NRPT.T.(Ref(normal_gbm_prior), xs)
end

function make_sampling_problem()
    problem = GBMProblem(normal_gbm_prior, NormalLikelihood())
end

function make_path(order::Union{Int, Nothing})
    if isnothing(order)
        return LinearPath()
    elseif order == 0
        # Linear path on untransformed space
        return ScalingGBMPath(1, LinearPath(), AutoForwardDiff())
    else 
        return ScalingGBMPath(order, LinearPath(), AutoForwardDiff())
    end
end

const c0 = 10.182337649086286 # This ensures that \delta_0 = 0.9

function make_optimizer(order::Union{Int, Nothing})
    if isnothing(order) || order == 0
        return NoOptState()
    else
        return NewtonTrustRegionState(AutoForwardDiff(), DecayrESSCriterion(c0, 0.5); prox=LowerBound([-Inf; zeros(order-1)]))
    end
end

function make_config(problem, order::Union{Int, Nothing})
    path = make_path(order)
    ptproblem = PathProblem(problem, path, SliceSampler())
    optimizer = make_optimizer(order)
    return NRPTConfig(
        ptproblem, optimizer;
        n_chains = N_CHAINS,
        n_rounds = N_ROUNDS,
        min_steps_for_opt = 100,
        steps_per_round = n -> 2^n,
        threaded = true,
        record_samples = false,
    )
end

directory() = joinpath(@__DIR__, "normal-results-50-sigma-$N_ROUNDS-rounds-05-19")

function run_all(; runs = Dict())
    problem = make_sampling_problem()

    if !("SKL-no_opt" in keys(runs))
        @info "Running LinearPath baseline"
        runs["SKL-no_opt"] = optimized_nrpt(make_config(problem, nothing))
    else
        @info "Found SKL-no_opt, skipping..."
    end

    for order in ORDERS
        name = "SKL-gbm-$order"
        if !(name in keys(runs))
            @info "Running ScalingGBMPath(order=$order)"
            runs[name] = optimized_nrpt(make_config(problem, order))
        else
            @info "Found $name, skipping..."
        end
    end

    mkpath(directory())
    return runs
end

function show_results(runs; dir = directory())
    mkpath(dir)
    colors = make_colors()
    name_map = make_name_map()
    title = "Conjugate Gaussian"

    p = compare_barriers(title, runs, colors, name_map;
            sch_tick = 3,
            opt_tick = 2,
            all_tick = 3
    )
    println(p)
    save(joinpath(dir, "barriers.pdf"), p)

    p = compare_cumulative_barriers(title, runs, colors, name_map)
    save(joinpath(dir, "cumulative-barriers.pdf"), p)

    p = compare_params(title, runs, colors, name_map)
    save(joinpath(dir, "param-evolution.pdf"), p)

    Λ_linear = runs["SKL-no_opt"].schedule_recorder.Λ_rej[end]
    E_linear = runs["SKL-no_opt"].schedule_recorder.Λ_acc[end]
    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=Λ_linear, ineff=E_linear)
    save(joinpath(dir, "cumulative-round-trips-iter-ineff-asympt.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=Λ_linear, ineff=nothing)
    save(joinpath(dir, "cumulative-round-trips-iter-asympt.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=nothing, ineff=E_linear)
    save(joinpath(dir, "cumulative-round-trips-iter-ineff.pdf"), p)

    p = cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=nothing, ineff=nothing)
    save(joinpath(dir, "cumulative-round-trips-iter.pdf"), p)

    p = show_τ_paths(title, runs, colors, name_map)
    save(joinpath(dir, "tau-paths.pdf"), p)
    p = show_τ_paths(title, runs, colors, name_map; yscale=log10, ylims=nothing)
    save(joinpath(dir, "tau-paths-log.pdf"), p)

    fn = joinpath(dir, "opt_params.jld2")
    write_opt_params(runs, fn)

    # for name in keys(runs)
    #     dense_dir = joinpath(dir, "densities-$name")
    #     mkpath(dense_dir)
    #     for i in 1:1
    #         xlabel = "x"
    #         p = plot_density(title, runs, name, N_CHAINS, name_map, xlabel; dim=i)
    #         save(joinpath(dense_dir, "last_round_density-$name-dim-$i.pdf"), p)
    #     end
    # end
    return 
end

