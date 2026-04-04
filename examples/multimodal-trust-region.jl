using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2, LinearAlgebra

include("helpers/select-colors.jl")
include("helpers/create-optimizer-dict.jl")
include("helpers/plotting.jl")

# Number of perturbation terms per direction (PerturbedLinearPathBidir uses 2×N_TERMS params)
const N_TERMS   = 1
const DIMENSION = 5  # 2^DIMENSION modes

# Multimodal target: equal-weight mixture of 2^d Gaussians at corners of {-MU,+MU}^d.
# Reference: broad isotropic Gaussian centred at 0.
const MU       = 3.0   # half-distance between opposite corners
const SIGMA    = 0.3   # within-mode std dev
const SIGMA_REF = 0.3  # reference std dev (must cover all modes)

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
        :steps_per_round => 1000,
        :n_rounds        => 200,
        :n_chains        => 9,
        :max_tr_steps    => 1,   # TR does this many IS-reweighted steps per round;
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
make_x0(n_chains) = [zeros(DIMENSION) for _ in 1:n_chains]

# Build 2^d corners of {-MU, +MU}^d.  Corner k corresponds to the binary
# representation of k: bit j = 0 → -MU, bit j = 1 → +MU.
function corner_means(d, mu)
    [Float64[((k >> (j-1)) & 1) == 1 ? mu : -mu for j in 1:d] for k in 0:(2^d - 1)]
end

function make_sampling_problem()
    D0    = MvNormal(zeros(DIMENSION), SIGMA_REF^2 * I)
    means = corner_means(DIMENSION, MU)
    D1    = MixtureModel(MvNormal[MvNormal(μ, SIGMA^2 * I) for μ in means])
    return GenericDistributionProblem(D0, D1)
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
    k         = pt_config.max_tr_steps
    is_newton = typeof(opt_state) <: NewtonTrustRegionState

    if is_newton
        # More samples per round, fewer rounds: Newton-TR round does k IS-reweighted
        # Newton steps on k× the data, matching the plain optimizer's total budget.
        optimizer     = opt_state
        n_rounds      = pt_config.n_rounds ÷ k
        steps_per_rnd = pt_config.steps_per_round * k
    elseif optimized && use_trust_region
        optimizer     = TrustRegionState(ProximalStochOptState(opt_state, make_prox_opt()); max_steps=k)
        n_rounds      = pt_config.n_rounds
        steps_per_rnd = pt_config.steps_per_round
    else
        optimizer     = make_optimizer(opt_state)
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

directory(config) = "multimodal-transport"

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
        nt = (
            x              = run.x,
            schedules      = run.schedules,
            Λ_rej          = run.Λ_rej,
            Λ_acc          = run.Λ_acc,
            objective_vals = run.objective_vals,
            index_process  = run.index_process,
            rejections     = run.rejections,
            logZsf         = run.logZsf,
            logZsb         = run.logZsb,
        )
        if hasproperty(run, :opt_state) && hasproperty(run.opt_state, :min_eigvals)
            nt = merge(nt, (; min_eigvals = copy(run.opt_state.min_eigvals)))
        end
        filtered[name] = nt
    end
    return filtered
end

function setup_optimizers(config)
    optimizers = make_optimizers(config.optimizer_config)
    colors     = color_per_opt(optimizers, config.optimizer_colors)
    return optimizers, colors
end

function show(config, runs, colors)
    dir      = directory(config)
    n_rounds = config.pt_config.n_rounds
    n_chains = config.pt_config.n_chains
    steps    = config.pt_config.steps_per_round

    p = compare_barriers(runs, n_rounds, colors; window_size=1)
    Plots.savefig(p, joinpath(dir, "barriers.svg"))

    p = compare_barriers(runs, n_rounds, colors; window_size=20)
    Plots.savefig(p, joinpath(dir, "barriers-smoothed.svg"))

    p = compare_params(runs, colors)
    Plots.savefig(p, joinpath(dir, "param-evolution.svg"))

    p = compare_min_eigvals(runs, colors)
    Plots.savefig(p, joinpath(dir, "min-eigvals.svg"))

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
