using Distributions, StatsPlots, DifferentiationInterface, Mooncake, NRPT, Random, ColorSchemes
using ForwardDiff
using Printf, LogExpFunctions

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

# This file compares tempering between two d
output_dir = "gauss-results/"
mkpath(output_dir)

D0 = Normal(0, 1)
sample_iid() = rand(D0)
V0(x) = -logpdf(D0, x)
V(x, d) = -logpdf(Normal(d, 1), x) + 100
data = [10.]

n_chains = 10
n_rounds = 1000

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

function run_spline_path!(runs, name, optimizer, n_knots)
    # Run the same problem with NRPT
    opt_state = (typeof(optimizer) != NoOptState
        ? ProximalStochOptState(optimizer, ProjectionState(Box(-10ones(2n_knots), 10ones(2n_knots))))
        : optimizer)
    println(opt_state)
    sampling_problem = NormalProblem(-1., 0.01, 1., 0.01)
    p_path = SplinePath(n_knots, 1., sampling_problem, AutoForwardDiff())
    ptproblem = PathProblem(p_path, NormalIIDExplorer())
    init_schedule = collect(range(0, 1, n_chains))
    x0 = ones(n_chains)

    runs[name] = optimized_nrpt(
        x0, init_schedule, ptproblem, opt_state;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 100,
    )
end

optimizers = Dict([
    "DoG" => DoGState(1e-6, 1e-6),
    "DoWG" => DoWGState(1e-6, 1e-6),
    "Adam-2" => AdamState(2., 1e-6),
    "Adam-0.2" => AdamState(0.2, 1e-6),
    "Adam-0.02" => AdamState(0.02, 1e-6),
    "Adam-0.002" => AdamState(0.002, 1e-6),
    "AdaGrad-2" => AdagradState(2., 1e-6),
    "AdaGrad-0.2" => AdagradState(0.2, 1e-6),
    "AdaGrad-0.02" => AdagradState(0.02, 1e-6),
    "AdaGrad-0.002" => AdagradState(0.002, 1e-6),
    "no_opt" => NoOptState(),
])

function run_comparison(; runs = Dict())
    # nrpt_linear = run_linear_NRPT()
    # runs["linear"] = nrpt_linear
    n_knots = 5
    for (name, optimizer) in optimizers
        @info "Running NRPT with '$name' optimizer "
        if !(name in keys(runs))
            run_spline_path!(runs, name, optimizer, n_knots)
        end
    end
    return runs
end

function compare_barriers(runs; window_size=10)
    colors = palette(:seaborn_dark, length(runs))
    p1 = plot(title="Barrier comparison", yscale=:log, legend=:outertopright)
    p2 = plot(title="SKL comparison", yscale=:log, legend=:outertopright)
    p3 = plot(title="Round trips", legend=:outertopright)
    for (c, (name, run)) in zip(colors, runs)
        plot!(p1, moving_average([b(1.0) for b in run.barriers], window_size), label="$name", color=c)
        if length(run.SKL_ests) > 0 && run.SKL_ests[1] !== nothing
            plot!(p2, moving_average(run.SKL_ests, window_size), label="$name", color=c)
        end
        rts = count_round_trips_per_round(run.index_process, n_rounds)
        plot!(p3, moving_average(rts, window_size), label="$name", color=c)
    end
    return plot(p1, p2, p3, layout=@layout([a; b; c]), plot_title="Optimization comparison")
end

function compare_barriers(runs, color_map; window_size=10)
    p1 = plot(title="Barrier comparison", yscale=:log, legend=:outertopright)
    p2 = plot(title="SKL comparison", yscale=:log, legend=:outertopright)
    p3 = plot(title="Round trips", legend=:outertopright)
    for (name, run) in runs
        c = color_map[name]
        plot!(p1, moving_average([b(1.0) for b in run.barriers], window_size), label="$name", color=c)
        if length(run.SKL_ests) > 0 && run.SKL_ests[1] !== nothing
            plot!(p2, moving_average(run.SKL_ests, window_size), label="$name", color=c)
        end
        rts = count_round_trips_per_round(run.index_process, n_rounds)
        plot!(p3, moving_average(rts, window_size), label="$name", color=c)
    end
    return plot(p1, p2, p3, layout=@layout([a; b; c]), plot_title="Optimization comparison")
end



function plot_density(runs, name, inds)
    run = runs[name]
    p = density(run.x[:, end-inds:end]', palette=:RdBu, linewidth=2, title="Density plot $name", label=reshape([@sprintf("β = %.3f", f) for f in run.schedules[:, end]] , 1, n_chains))
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
    p1 = plot(title="Final barrier comparison", legend=:outertopright)
    βs = 0.0:0.0001:1.0
    for (c, (name, run)) in zip(colors, runs)
        plot!(p1, βs, x -> run.barriers[end](x), label="$name", color=c)
        scatter!(p1, run.schedules[:, end], x -> run.barriers[end](x), label="$name", color=c)
    end
    return p1
end

function compare_cumulative_barrier_evolution(runs, name, inds)
    pal=palette(:RdBu, length(inds))
    p1 = plot(title="Cumulative barrier evo. $name", legend=:outertopright)
    βs = 0.0:0.01:1.0
    run = runs[name]
    palette
    for (j, i) in enumerate(inds)
        barrier = run.barriers[i]
        plot!(p1, βs, x -> barrier(x), label="$name", color=pal[j])
        scatter!(p1, run.schedules[:, i], x -> barrier(x), label="Iteration $i", color=pal[j])
    end
    return p1
end




function compare_rt_barrier(runs)
    vruns = collect(runs)
    rts = [round_trip_rate(run.index_process) for (_, run) in vruns]
    rt_ests = [1 / (1 + 2run.barriers[end](1.0)) for (_, run) in vruns]
    data = hcat(rts, rt_ests)
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
        if !(startswith(name, "no-opt"))
            plot!(p1, stack(run.opt_state.xs)', label="$name", ylabel="q")
            plot!(p2, run.opt_state.etas, label="$name", ylabel="η")
        end
    end
    return plot(p1, p2, layout=@layout([a; b]))
end

function plot_paths(runs, name, inds)
    p = plot(palette=palette(:RdBu, length(inds)))
    for i in inds
        eta = theta_to_eta(runs[name].opt_state.xs[i], [false, true])
        plot!(eta[1, :], eta[2, :])
    end
    p
end

function plot_paths2(runs, name, inds)
    pal1 = palette(:reds, length(inds)+3; rev=true)
    pal2 = palette(:blues, length(inds)+3; rev=true)
    p = plot(title="Splines $name", legend=:outertopright)
    for (j, i) in enumerate(inds)
        eta = theta_to_eta(runs[name].opt_state.xs[i], [false, true])
        plot!(eta[2, :], c=pal1[j], label="η_1")
        plot!(eta[1, :], c=pal2[j], label="η_2")
    end
    p
end

function plotZ(runs, pal=:seaborn_bright)
    colors = palette(pal, length(runs))
    p = plot(title="Normalization constant")
    for (c, (name, run)) in zip(colors, runs)
        mZ = (run.logZsf + run.logZsb) / 2
        plot!(p, run.logZsf, label="$name (f)", ylabel="Z", alpha=0.3, color=c)
        plot!(p, run.logZsb, label="$name (b)", ylabel="Z", alpha=0.3, color=c)
        plot!(p, mZ, label="$name (average)", ylabel="Z", color=c)
    end
    plot!(p, 1:1000, -100 * ones(1000), label="true Z", color=:black)
    return p
end

function plot_all(runs, dir)
    p = compare_barriers(runs; window_size=1)
    savefig(p, joinpath(dir, "barriers.png"))
    p = compare_barriers(runs; window_size=20)
    savefig(p, joinpath(dir, "barriers-20.png"))

    for (name, _) in runs in 
        if !(startswith(name, "no-opt"))
            p = plot_paths2(runs, name, 1000:1000)
            savefig(p, joinpath(dir, "splines-$name.png"))
        end
        p = compare_cumulative_barrier_evolution(runs, name, 1:100:1000)
        savefig(p, joinpath(dir, "cbe-$name.png"))
        p, _ = plot_density(runs, name, 1000)
        savefig(p, joinpath(dir, "final-density-$name.png"))
    end 

end