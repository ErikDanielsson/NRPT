using Distributions, StatsPlots, DifferentiationInterface, Mooncake, NRPT, Random
using Pigeons, MCMCChains

# This file compares tempering between two d
output_dir = "pigeons-vs-nrpt-results/"
mkpath(output_dir)

D0 = Normal(0, 1)
sample_iid() = rand(D0)
V0(x) = -logpdf(D0, x)
V(x, d) = -logpdf(Normal(d, 0.1), x)
data = [10.]
sampling_problem = SamplingProblem(V0, sample_iid, V, data)

n_chains = 10

# Run with Pigeons first to get an estimate of Λ
struct GaussLogPotential
    data::Vector{Float64}
    D0::Distribution
end

function (log_potential::GaussLogPotential)(x)
    return -(V0(x[1]) + sum(V(x[1], d) for d in log_potential.data; init=0.0))
end

Pigeons.initialization(::GaussLogPotential, ::AbstractRNG, ::Int) = [1.0]

function Pigeons.sample_iid!(lp::GaussLogPotential, replica, shared)
    state = replica.state
    rng = replica.rng
    state[1] = rand(rng, lp.D0)
end

n_rounds = 15
function run_pigeons()
    return pigeons(
        n_chains = n_chains,
        n_rounds = n_rounds,
        target=GaussLogPotential(data, D0),
        reference=GaussLogPotential([], D0),
        record=[round_trip; record_default()] 
    )
end

function run_NRPT()
    # Run the same problem with NRPT
    l_path = NRPT.linear_path(sampling_problem)
    ptproblem = PathProblem(l_path, IterExplorer(NRPT.SliceSampler(10., 3), 1))
    init_schedule = collect(range(0, 1, n_chains))
    x0 = ones(n_chains)


    return nrpt(
        x0, init_schedule, ptproblem;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 2^n,
    )
end

function compare_schedules(pt, nrpt_schedule)
    plot(title="Schedule comparison")
    plot(pt.shared.tempering.schedule.grids, label="Pigeons' schedule")
    plot!(nrpt_schedule, label="NRPT schedule")
end

function compare_final_barrier(pt, nrpt_barrier)
    βs = 0.0:0.001:1
    plot(title="Cumulative barrier comparison")
    plot!(βs, pt.shared.tempering.communication_barriers.cumulativebarrier.(βs), label="Pigeons")
    plot!(βs, nrpt_barrier.(βs), label="NRPT")
end

function compare_barrier_evolution(pt, nrpt_barriers)
    βs = 0.0:0.001:1
    summary =  pt.shared.reports.summary
    plot(title="Global barrier comparison")
    plot!(summary.global_barrier, label="Pigeons")
    plot!([b(1.0) for b in nrpt_barriers], label="NRPT")
end

function compare_rt_barrier(pt, nrpt_barrier, nrpt_index_process)
    summary =  pt.shared.reports.summary
    pigeons_rt = sum(summary.n_tempered_restarts) / sum(summary.n_scans)
    pigeons_rt_est = 1 / (1 + 2summary.global_barrier[end])
    nrpt_rt = round_trip_rate(nrpt_index_process)
    nrpt_rt_est = 1 / (1 + 2nrpt_barrier(1.0))
    data = [pigeons_rt pigeons_rt_est; nrpt_rt nrpt_rt_est]
    labels = vec(["τ" "(2Λ + 1)^(-1)" "τ" "(2Λ + 1)^(-1)"])
    names = vec(["Pigeons" "Pigeons" "NRPT" "NRPT"])
    return groupedbar(labels, data', group=names, bar_position = :dodge, bar_width = 0.7, title="Round trip and estimator comparison")  
end

function run_and_compare()
    x, schedules, barriers, index_process= run_NRPT()
    pt = run_pigeons()

    p1 = compare_schedules(pt, schedules[:, end])
    savefig(p1, joinpath(output_dir, "schedule-comparison.png"))

    p2 = compare_final_barrier(pt, barriers[end])
    savefig(p2, joinpath(output_dir, "cumulative-barrier-comparison.png"))

    p3 = compare_barrier_evolution(pt, barriers)
    savefig(p3, joinpath(output_dir, "global-barrier-evolution-comparison.png"))

    p4 = compare_rt_barrier(pt, barriers[end], index_process)
    savefig(p4, joinpath(output_dir, "round-trip-comparison.png"))
end