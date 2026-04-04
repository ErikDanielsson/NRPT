using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2, LinearAlgebra, DelimitedFiles

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

# Number of perturbation terms; c = 4·softmax(t) so Σcᵢ = 4 automatically
const N_TERMS = 1
const DIMENSION = 5  # [log10(km0), log10(δ), log10(β), log10(t0), log10(σ)]

# Prior bounds in log10 space for each parameter
#   km0, δ, β  ~ LogUniform(-5, 5)
#   t0          ~ LogUniform(-2, 1)
#   σ           ~ LogUniform(-2, 2)
const PRIOR_LB = [-5.0, -5.0, -5.0, -2.0, -2.0]
const PRIOR_UB = [ 5.0,  5.0,  5.0,  1.0,  2.0]

# ODE mean: km0/(δ-β)·(1 - exp(-(δ-β)·(t-t0)))·exp(-β·(t-t0))
# Matches the reference implementation exactly. NaN/Inf is handled in the
# likelihood by clamping to 10_000 (a hack for vague priors, kept for replication).
function ode_mean(t, km0, δ, β, t0)
    Δ = δ - β
    return km0 / Δ * (1 - exp(-Δ * (t - t0))) * exp(-β * (t - t0))
end

# Build a PosteriorProblem from a data matrix whose columns are [t, O_t].
# Works with any 2-column array (N×2 matrix or vector of 2-element vectors).
function make_sampling_problem(data_matrix::AbstractMatrix)
    # Convert rows to (t, O_t) tuples stored as a Vector
    data = [(data_matrix[i, 1], data_matrix[i, 2]) for i in axes(data_matrix, 1)]

    # Log prior: uniform on log10 scale — constant within box, -Inf outside
    function log_prior(x)
        for i in eachindex(x)
            (x[i] < PRIOR_LB[i] || x[i] > PRIOR_UB[i]) && return -Inf
        end
        return zero(eltype(x))
    end

    # IID sampler from the reference (prior)
    function sample_prior()
        return [rand() * (PRIOR_UB[i] - PRIOR_LB[i]) + PRIOR_LB[i] for i in 1:DIMENSION]
    end

    # Log likelihood for a single observation d = (t, O_t)
    # x = [log10(km0), log10(δ), log10(β), log10(t0), log10(σ)]
    # Variance of the Normal is σ², so std dev passed to Normal() is σ = 10^x[5].
    function log_likelihood(x, d)
        t, O = d
        km0 = 10^x[1]
        δ   = 10^x[2]
        β   = 10^x[3]
        t0  = 10^x[4]
        σ   = 10^x[5]
        μ   = ode_mean(t, km0, δ, β, t0)
        if isnan(μ) || isinf(μ)
            μ = 10_000.0   # hack: priors too vague but keeping them for replication
        end
        return logpdf(Normal(μ, σ), O)
    end

    return PosteriorProblem(log_prior, sample_prior, log_likelihood, data)
end

function make_config()
    optimizer_config = Dict([
        "DoG"    => nothing,
        "no_opt" => nothing,
    ])

    optimizer_colors = Dict(
        "DoWG"      => colorant"#d62728",
        "DoWG-TR"   => colorant"#ff7f0e",
        "DoG"       => colorant"#35912f",
        "DoG-TR"    => colorant"#17becf",
        "no_opt"    => colorant"#9e9e9e",
    )

    pt_config = NamedTuple([
        :steps_per_round => 100,
        :n_rounds        => 10,
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

function make_prox_opt()
    return NoProx()
end

function make_optimizer(opt_state)
    return (typeof(opt_state) != NoOptState
        ? ProximalStochOptState(opt_state, make_prox_opt())
        : opt_state)
end

function run_problem!(runs, name, opt_state, pt_config, data_matrix; use_trust_region=false)
    optimized = typeof(opt_state) != NoOptState
    problem   = make_sampling_problem(data_matrix)
    path      = make_path(optimized)
    ptproblem = make_path_problem(problem, path)
    optimizer = make_optimizer(opt_state)
    if optimized && use_trust_region
        optimizer = TrustRegionState(optimizer)
    end

    n_chains      = pt_config.n_chains
    x0            = make_x0(n_chains)
    init_schedule = make_init_schedule(n_chains)

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
        for (key, c) in optimizer_colors
            if endswith(name, key)
                colors[name] = c
                break
            end
        end
        if !haskey(colors, name)
            colors[name] = colorant"#000000"
        end
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

directory(config) = "transfection-ode"

function run(config, data_matrix; runs=Dict(), run_fn="run.jld2", config_fn="config.jld2")
    optimizers, colors = setup_optimizers(config)
    for (name, (_, opt_state)) in optimizers
        is_no_opt = typeof(opt_state) == NoOptState
        variants = is_no_opt ? [(name, false)] : [(name, false), ("$name-TR", true)]
        for (run_name, use_tr) in variants
            if !(run_name in keys(runs))
                @info "Running NRPT with $run_name"
                run_problem!(runs, run_name, opt_state, config.pt_config, data_matrix;
                             use_trust_region=use_tr)
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

# Load transfection data from processed.csv (columns: times, sample, observations).
# Returns an N×2 matrix [times observations].
function load_transfection_data(path=joinpath(@__DIR__, "transfection", "processed.csv"))
    raw = readdlm(path, ',', Float64; skipstart=1)  # skip header row
    return hcat(raw[:, 1], raw[:, 3])               # columns: times, observations
end

# Example usage:
#
#   data = load_transfection_data()
#   config = make_config()
#   runs, colors = run(config, data)
#   show(config, runs, colors)
