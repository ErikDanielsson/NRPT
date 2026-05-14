struct NRPTConfig{T, S}
    x0::Vector{T}
    problem::PathProblem
    opt_state::Optimizer
    objective::PathObjective
    seed::Int
    threaded::Bool
    n_rounds::Int
    min_steps_for_opt::Int
    steps_per_round::S
    use_accept::Bool
    save_rejection::Bool
    save_lps::Bool
    record_samples::Bool
    progress::Bool
end

n_chains(config::NRPTConfig) = length(config.x0)
function Base.show(io::IO, config::NRPTConfig)
    n = n_chains(config)
    sample_iters = [config.steps_per_round(i) for i in 1:min(4, config.n_rounds)]
    last_iter = config.steps_per_round(config.n_rounds)
    total_iter = sum(config.steps_per_round, 1:config.n_rounds)
    iter_str = join(sample_iters, ", ") * (config.n_rounds > 4 ? ", …" : "") * ", " * "$last_iter"
    println(io, "NRPTConfig:")
    println(io, "  problem:           $(typeof(config.problem))")
    println(io, "  typeof(x0):        $(typeof(config.x0[1]))")
    println(io, "  n_chains:          $n")
    println(io, "  n_rounds:          $(config.n_rounds)")
    println(io, "  threaded:          $(config.threaded)")
    println(io, "  min_steps_for_opt: $(config.min_steps_for_opt)")
    println(io, "  steps_per_round:   [$iter_str] ($total_iter)")
    println(io, "  opt_state:         $(typeof(config.opt_state))")
    println(io, "  objective:         $(typeof(config.objective))")
    println(io, "  use_accept:        $(config.use_accept)")
    println(io, "  seed:              $(config.seed)")
    return println(io, "  progress:          $(config.progress)")
end

# Constructor 1: explicit x0
function NRPTConfig(
        x0::Vector{T},
        problem::PathProblem,
        opt_state::Optimizer = NoOptState();
        seed::Int = 2,
        threaded::Bool = true,
        n_rounds::Int = 10,
        min_steps_for_opt::Int = 100,
        steps_per_round = n -> 2^n,
        use_accept::Bool = false,
        save_rejection::Bool = false,
        save_lps::Bool = false,
        record_samples::Bool = true,
        objective::PathObjective = SKLObjective(),
        progress::Bool = true,
    ) where {T}
    return NRPTConfig{T, typeof(steps_per_round)}(
        x0,
        problem,
        opt_state,
        objective,
        seed,
        threaded,
        n_rounds,
        min_steps_for_opt,
        steps_per_round,
        use_accept,
        save_rejection,
        save_lps,
        record_samples,
        progress
    )
end

# Constructor 2: make_x0(i) factory
function NRPTConfig(
        make_x0,
        problem::PathProblem,
        opt_state::Optimizer = NoOptState();
        n_chains::Int = 10,
        kwargs...
    )
    x0 = [make_x0(i) for i in 1:n_chains]
    return NRPTConfig(x0, problem, opt_state; kwargs...)
end

# Constructor 3: no x0, sample IID from problem
function NRPTConfig(
        problem::PathProblem,
        opt_state::Optimizer = NoOptState();
        n_chains::Int = 10,
        kwargs...
    )
    x0 = [sample_iid(problem.problem) for _ in 1:n_chains]
    return NRPTConfig(x0, problem, opt_state; kwargs...)
end
