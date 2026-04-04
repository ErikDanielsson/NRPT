using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2, LinearAlgebra

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

# Number of perturbation terms
const N_TERMS = 1
const DIMENSION = 2  # [x, y] ∈ (0,1)²

# Unidentifiable product model:
#   nf | nt, x, y ~ Binomial(nt, x·y)
#   X, Y ~ Uniform(0, 1)
#   nt = 100_000, nf = 50_000
# Posterior concentrates on the curve x·y = 0.5 in [0,1]².
const NT = 100_000
const NF = 50_000

function make_config()
    optimizer_config = Dict([
        "DoG"    => nothing,
        "no_opt" => nothing,
    ])

    optimizer_colors = Dict(
        "DoWG"      => colorant"#d62728",  # red       — no trust region
        "DoWG-TR"   => colorant"#ff7f0e",  # orange    — trust region
        "DoG"       => colorant"#35912f",  # green     — no trust region
        "DoG-TR"    => colorant"#17becf",  # cyan      — trust region
        "no_opt"    => colorant"#9e9e9e",  # gray      — LinearPath baseline
    )

    pt_config = NamedTuple([
        :steps_per_round => 100,
        :n_rounds        => 100,
        :n_chains        => 100,
        :max_tr_steps    => 10,   # TR does this many IS-reweighted steps per round;
                                  # TR rounds  = n_rounds / max_tr_steps  (÷ max_tr_steps)
                                  # TR spr     = steps_per_round * max_tr_steps (× max_tr_steps)
                                  # → same total MCMC budget, same total optimizer steps
    ])

    return NamedTuple([
        :pt_config         => pt_config,
        :optimizer_config  => optimizer_config,
        :optimizer_colors  => optimizer_colors,
    ])
end

make_init_schedule(n_chains) = collect(range(0, 1, n_chains))
make_x0(n_chains) = [fill(0.5, DIMENSION) for _ in 1:n_chains]

function make_sampling_problem()
    # Single "observation": (nt, nf)
    data = [(NT, NF)]

    # Log prior: Uniform(0,1)² — constant within unit square, -Inf outside
    function log_prior(x)
        for i in eachindex(x)
            (x[i] <= 0.0 || x[i] >= 1.0) && return -Inf
        end
        return zero(eltype(x))
    end

    function sample_prior()
        return rand(DIMENSION)
    end

    # Log likelihood: nf ~ Binomial(nt, x·y)
    function log_likelihood(x, d)
        nt, nf = d
        p = x[1] * x[2]
        # Clamp to valid probability (defensive; prior already excludes boundary)
        p = clamp(p, eps(), 1.0 - eps())
        return logpdf(Binomial(nt, p), nf)
    end

    return PosteriorProblem(log_prior, sample_prior, log_likelihood, data)
end

function make_path(optimized::Bool)
    if optimized
        return PerturbedLinearPathBidir(N_TERMS, AutoForwardDiff())
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

function run_problem!(runs, name, opt_state, pt_config; use_trust_region=false)
    optimized = typeof(opt_state) != NoOptState
    problem   = make_sampling_problem()
    path      = make_path(optimized)
    ptproblem = make_path_problem(problem, path)
    optimizer = make_optimizer(opt_state)

    k = pt_config.max_tr_steps
    if optimized && use_trust_region
        # More samples per round, fewer rounds: each TR round does k IS-reweighted
        # gradient steps on k× the data, matching the plain optimizer's total budget.
        optimizer     = TrustRegionState(optimizer, max_steps=k)
        n_rounds      = pt_config.n_rounds ÷ k
        steps_per_rnd = pt_config.steps_per_round * k
    else
        n_rounds      = pt_config.n_rounds
        steps_per_rnd = pt_config.steps_per_round
    end

    n_chains      = pt_config.n_chains
    x0            = make_x0(n_chains)
    init_schedule = make_init_schedule(n_chains)

    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=_ -> steps_per_rnd,
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

directory(config) = "unidentifiable-product"

function run(config; runs=Dict(), run_fn="run.jld2", config_fn="config.jld2")
    optimizers, colors = setup_optimizers(config)
    for (name, (_, opt_state)) in optimizers
        is_no_opt = typeof(opt_state) == NoOptState
        variants = is_no_opt ? [(name, false)] : [(name, false), ("$name-TR", true)]
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

    p = cumulative_round_trips(runs, n_rounds, colors)
    Plots.savefig(p, joinpath(dir, "cumulative-round-trips.svg"))

    for (name, _) in runs
        p = plot_density(runs, name, 10 * steps, n_chains; dim=1)
        Plots.savefig(p, joinpath(dir, "density-$name.svg"))
    end

    return runs, colors
end

# Example usage:
#
#   config = make_config()
#   runs, colors = run(config)
#   show(config, runs, colors)
