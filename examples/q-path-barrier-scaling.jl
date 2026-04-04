using Distributions, StatsPlots, DifferentiationInterface, NRPT, Random, ColorSchemes
using ForwardDiff, JLD2

# Experiment: how does the tempering barrier Λ_rej scale with number of chains?
# Problem: two normal distributions, QPath with q=0.5

function make_sampling_problem()
    return GenericDistributionProblem(Normal(0.0, 1.0), Normal(10.0, 1.0))
end

function make_path_problem()
    sprob = make_sampling_problem()
    path = QPath(0.5, AutoForwardDiff())
    PathProblem(sprob, path, IterExplorer(SliceSampler(), 100))
end

make_init_schedule(n_chains) = collect(range(0.0, 1.0, n_chains))
make_x0(n_chains) = zeros(n_chains)

function run_for_n_chains(n_chains; n_rounds=200, steps_per_round=100)
    problem = make_path_problem()
    x0 = make_x0(n_chains)
    schedule = make_init_schedule(n_chains)
    result = optimized_nrpt(
        x0, schedule, problem, NoOptState();
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=_ -> steps_per_round,
        objective=BarrierObjective()
    )
    return result
end

function run_scaling_experiment(n_chains_list; n_rounds=200, steps_per_round=100)
    results = Dict{Int, Any}()
    for n in n_chains_list
        @info "Running with n_chains = $n"
        results[n] = run_for_n_chains(n; n_rounds=n_rounds, steps_per_round=steps_per_round)
    end
    return results
end

# Extract the final (time-averaged) Λ_rej from a run
function final_barrier(result; last_frac=0.5)
    Λ = result.Λ_rej
    start = max(1, round(Int, (1 - last_frac) * length(Λ)))
    return mean(Λ[start:end])
end

predicted_rtr(Λ) = 1.0 / (1.0 + 2.0 * Λ)

function plot_barrier_scaling(results, n_chains_list)
    barriers = [final_barrier(results[n]) for n in n_chains_list]
    p = plot(
        n_chains_list, barriers;
        xlabel="Number of chains N",
        ylabel="Λ_rej",
        title="Tempering barrier vs number of chains\n(QPath q=0.5, Normal(0,1) → Normal(5,1))",
        marker=:circle,
        linewidth=2,
        label="Λ_rej (mean over last 50% of rounds)",
        legend=:topright
    )
    return p
end

function plot_barrier_traces(results, n_chains_list)
    p = plot(
        xlabel="Round",
        ylabel="Λ_rej",
        title="Barrier traces by number of chains\n(QPath q=0.5, Normal(0,1) → Normal(5,1))"
    )
    colors = cgrad(:viridis, length(n_chains_list); categorical=true)
    for (i, n) in enumerate(n_chains_list)
        plot!(p, results[n].Λ_rej; label="N=$n", color=colors[i], linewidth=1.5, alpha=0.8)
    end
    return p
end

function plot_rtr_comparison(results, n_chains_list)
    barriers  = [final_barrier(results[n]) for n in n_chains_list]
    emp_rtrs  = [round_trip_rate(results[n].index_process) for n in n_chains_list]
    pred_rtrs = predicted_rtr.(barriers)

    p = plot(
        xlabel="Number of chains N",
        ylabel="Round-trip rate",
        title="Empirical vs predicted round-trip rate\n(predicted: 1 / (1 + 2Λ_rej))",
        legend=:topright
    )
    plot!(p, n_chains_list, emp_rtrs;
        marker=:circle, linewidth=2, label="Empirical RTR")
    plot!(p, n_chains_list, pred_rtrs;
        marker=:diamond, linewidth=2, linestyle=:dash, label="1 / (1 + 2Λ_rej)")
    return p
end

const RESULTS_FILE = "q-path-barrier-scaling.jld2"
const N_CHAINS_LIST = [5, 10, 20, 50]

function run(; n_rounds=10, steps_per_round=1000, force=false)
    if !force && isfile(RESULTS_FILE)
        @info "Loading saved results from $RESULTS_FILE"
        return load(RESULTS_FILE, "results")
    end
    results = run_scaling_experiment(N_CHAINS_LIST; n_rounds=n_rounds, steps_per_round=steps_per_round)
    jldsave(RESULTS_FILE; results)
    return results
end

function plot_all(results)
    p_scaling = plot_barrier_scaling(results, N_CHAINS_LIST)
    p_traces  = plot_barrier_traces(results, N_CHAINS_LIST)
    p_rtr     = plot_rtr_comparison(results, N_CHAINS_LIST)
    display(p_scaling)
    display(p_traces)
    display(p_rtr)
    return p_scaling, p_traces, p_rtr
end

# Uncomment to run:
# results = run()
# plot_all(results)

# To force a rerun:
# results = run(force=true)
