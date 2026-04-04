using Distributions, StatsPlots, DifferentiationInterface, Mooncake, NRPT, Random, ColorSchemes
using Printf, LogExpFunctions

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

# This file compares tempering between two d
output_dir = "gauss-results/"
mkpath(output_dir)

D0 = Normal(0, 1)
sample_iid() = rand(D0)
V0(x) = -logpdf(D0, x)
V(x, d) = -logpdf(Normal(d, 1), x)
data = [10.]

n_chains = 10
n_rounds = 300

function run_linear_NRPT()
    # Run the same problem with NRP
    sampling_problem = SamplingProblem(V0, sample_iid, V, data)
    l_path = NRPT.linear_path(sampling_problem)
    ptproblem = PathProblem(l_path, IterExplorer(NRPT.SliceSampler(10., 3), 1))
    init_schedule = collect(range(0, 1, n_chains))
    x0 = ones(n_chains)


    return optimized_nrpt(
        x0, init_schedule, ptproblem;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 10,
    )
end

function run_power_path!(runs, name, opt_state)
    # Run the same problem with NRPT
    println(opt_state)
    sampling_problem = SamplingProblem(V0, sample_iid, V, data)
    p_path = NRPT.power_path(1., 1., sampling_problem, AutoMooncake())
    ptproblem = PathProblem(p_path, IterExplorer(NRPT.SliceSampler(10., 3), 1))
    init_schedule = collect(range(0, 1, n_chains))
    x0 = ones(n_chains)
    runs[name] =  optimized_nrpt(
        x0, init_schedule, ptproblem, opt_state;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 100,
    )
end

function run_q_path!(runs, name, opt_state, q0)
    # Run the same problem with NRPT
    println(opt_state)
    sampling_problem = SamplingProblem(V0, sample_iid, V, data)
    p_path = NRPT.q_path(q0, 1., sampling_problem, AutoMooncake())
    ptproblem = PathProblem(p_path, IterExplorer(NRPT.SliceSampler(10., 3), 1))
    init_schedule = collect(range(0, 1, n_chains))
    x0 = ones(n_chains)

    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, opt_state;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 50,
    )
end

function run_comparison()
    runs = Dict()
    # nrpt_linear = run_linear_NRPT()
    # runs["linear"] = nrpt_linear
    optimizers = Dict([
        "DoG-0.99" => (0.99, DoGState(1e-6, 1e-6)),
        "DoWG-0.99" => (0.99, DoWGState(1e-6, 1e-6)),
        "no-opt-0.99" => (0.99, NoOptState()),
        "DoG-0.9" => (0.9, DoGState(1e-6, 1e-6)),
        "DoWG-0.9" => (0.9, DoWGState(1e-6, 1e-6)),
        "no-opt-0.9" => (0.9, NoOptState()),
        "DoG-0.5" => (0.5, DoGState(1e-6, 1e-6)),
        "DoWG-0.5" => (0.5, DoWGState(1e-6, 1e-6)),
        "no-opt-0.5" => (0.5, NoOptState()),
    ])
    for (name, (param0, optimizer)) in optimizers
        @info "Running NRPT with '$name' optimizer "
        proxgrad = (
            typeof(optimizer) != NoOptState
            ? ProximalStochOptState(optimizer, ProjectionState(Box(1., 10000.)))
            : optimizer
        )
        run_q_path!(runs, name, proxgrad, param0)
    end
    return runs
end

function compare_barriers(runs; window_size=10)
    colors = palette(:seaborn_dark, length(runs))
    p1 = plot(title="Barrier comparison", yscale=:log, legend=:outertopright)
    p2 = plot(title="SKL comparison", yscale=:log, legend=:outertopright)
    for (c, (name, run)) in zip(colors, runs)
        plot!(p1, moving_average([b(1.0) for b in run.barriers], window_size), label="$name", color=c)
        if length(run.SKL_ests) > 0 && run.SKL_ests[1] !== nothing
            plot!(p2, moving_average(run.SKL_ests, window_size), label="$name", color=c)
        end
    end
    return plot(p1, p2, layout=@layout([a; b]), plot_title="q-path opt")
end

function plot_density(runs, name)
    run = runs[name]
    p = density(run.x[:, end-3000:end]', palette=:RdBu, linewidth=2, title="Density plot $name", label=reshape([@sprintf("β = %.3f", f) for f in run.schedules[:, end]] , 1, n_chains))
    return p, "density-plot-$name.png"
end

function save_densities(runs, dir)
    for (name, _) in runs
        p, fn = plot_density(runs, name)
        savefig(p, joinpath(dir, fn))
    end
end

function compare_cumulative_barriers(runs)
    colors = palette(:seaborn_bright, length(runs))
    p1 = plot(title="Barrier comparison", legend=:outertopright)
    βs = 0.0:0.01:1.0
    for (c, (name, run)) in zip(colors, runs)
        plot!(p1, βs, x -> run.barriers[end](x), label="$name", color=c)
        scatter!(p1, run.schedules[:, end], x -> run.barriers[end](x), label="$name", color=c)
    end
    return p1
end



function compare_rt_barrier(runs)
    vruns = collect(runs)
    rts = [round_trip_rate(run.index_process) for (_, run) in vruns]
    rt_ests = [1 / (1 + 2run.barriers[end](1.0)) for (_, run) in vruns]
    data = hcat(rts, rt_ests)
    println(data)
    labels = [n  for n in ["τ", "(2Λ + 1)^(-1)"] for _ in vruns ]
    names = [name  for _ in 1:2 for (name, _) in vruns]
    println(labels)
    println(names)
    return groupedbar(names, data, group=labels, bar_position = :dodge, bar_width = 0.7, title="Round trip and estimator comparison, q-path")  
end

function compare_params(runs)
    p1 = plot(title="Parameter evolution", legend=:outertopright)
    p2 = plot(title="Step size evolution", yscale=:log10, legend=:outertopright)
    for (name, run) in runs
        if !(name in ["linear", "no-opt"])
            plot!(p1, logistic.(run.opt_state.xs), label="$name", ylabel="q")
            plot!(p2, run.opt_state.etas, label="$name", ylabel="η")
        end
    end
    return plot(p1, p2, layout=@layout([a; b]))
end

