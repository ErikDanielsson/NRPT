using NRPT, Plots, Printf, LaTeXStrings, KernelDensity

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

function compare_barriers(runs, colors; window_size=1)
    f = CairoMakie.Figure()
    p1 = CairoMakie.Axis(f[1, 1], title="Barrier comparison")
    p2 = CairoMakie.Axis(f[1, 2], title="SKL comparison")
    p3 = CairoMakie.Axis(f[2, 1], title="Round trip rate")
    p4 = CairoMakie.Axis(f[2, 2], title="Λ (opt rounds)")
    for (name, run) in runs
        c = colors[name]
        CairoMakie.lines!(p1, moving_average(run.schedule_recorder.Λ_rej, window_size), label="$name", color=c)
        obj = get_objective_vals(run.loss_recorder)
        if !isempty(obj)
            println(obj)
            println(max.(1e-8, moving_average(obj, window_size)))
            CairoMakie.lines!(p2, max.(1e-8, moving_average(obj, window_size)), label="$name", color=c)
        end
        rts   = count_round_trips_per_round(run.index_process)
        rates = rts ./ run.index_process.rounds
        CairoMakie.lines!(p3, moving_average(rates, window_size), label="$name", color=c)
        λs = get_Λ_opt_round(run.loss_recorder)
        if !isempty(λs)
            CairoMakie.lines!(p4, moving_average(λs, window_size), label="$name", color=c)
        end
    end
    return f
    return p1, p2, p3, p4# , CairoMakie.lines(p1, p2, p3, p4, layout=@layout([a b; c d]), lines_title="Optimization comparison", size=(1200, 800))
    # return CairoMakie.plot(p1, p3, p4, layout=@layout([a b; d]), plot_title="Optimization comparison", size=(1200, 800))
end

function cumulative_round_trips(runs, colors)
    p = StatsPlots.plot(title="Cumualtive round trips", legend=:outertopright)
    for (name, run) in runs
        c = colors[name]
        rts = cumsum(count_round_trips_per_round(run.index_process))
        StatsPlots.plot!(p, rts, label="$name", color=c)
    end
    return p
end

function plot_exponents(runs, colors; ind=nothing)
    βs = 0:0.01:1
    p = StatsPlots.plot()
    for (name, run) in runs
        e1s = Float64[]
        e2s = Float64[]
        for β in βs
            if !(typeof(run.opt_state) == NoOptState)
                if ind === nothing
                    NRPT.set_param!(run.problem.path, run.opt_state.xs[end])
                else
                    NRPT.set_param!(run.problem.path, run.opt_state.xs[ind])
                end
            end
            e1, e2 = get_exponents(run.problem.path, β)
            push!(e1s, e1)
            push!(e2s, e2)
        end
        StatsPlots.plot!(e1s, e2s, label="$name", color=colors[name])
    end
    p
end

function plot_exponents_log(runs, colors; ind=nothing, ϵ=1e-5)
    βs = ϵ:0.01:1.0 - ϵ
    p = StatsPlots.plot(size=(1200, 800), legend=:outertopright)
    for (name, run) in runs
        e1s = Float64[]
        e2s = Float64[]
        if !(typeof(run.opt_state) == NoOptState)
            if ind === nothing
                NRPT.set_param!(run.problem.path, run.opt_state.xs[end])
            else
                NRPT.set_param!(run.problem.path, run.opt_state.xs[ind])
            end
        end
        for β in βs
            e1, e2 = get_exponents(run.problem.path, β)
            push!(e1s, e1)
            push!(e2s, e2)
        end
        StatsPlots.plot!(e1s, e2s, label="$name", color=colors[name], yscale=:log10, xscale=:log10)
    end
    p
end

function plot_schedule(runs, name; kwargs...)
    run = runs[name]
    schedules = get_schedules(run.schedule_recorder)
    n_chains = length(schedules[:, 1])
    pal = palette(:bamako, n_chains - 1)
    p = StatsPlots.plot(
        palette=pal,
        linewidth=2,
        title="Schedule evolution - $name",
        legend=false,
        size=(1200, 800);
        kwargs...
    )
    StatsPlots.plot!(p, schedules[2:end, :]', yscale=:log10)
    return p
end

function plot_density(runs, name, n_chains; round=nothing, subsample_chains=1, dim=nothing, kwargs...)
    run = runs[name]
    pal = palette(:bamako, n_chains)
    n_rounds = length(run.x.rounds) - 1
    r = isnothing(round) ? n_rounds : round
    xs = get_round_samples(run.x, r)[1:subsample_chains:end, :]
    if eltype(xs) <: AbstractVector
        d = isnothing(dim) ? 1 : dim
        xs = map(x -> x[d], xs)
    end
    schedules = get_schedules(run.schedule_recorder)
    p = StatsPlots.plot(
        palette=pal,
        linewidth=2,
        title="Density plot $name (round $r)",
        legend=:outertopright,
        size=(1200, 800);
        kwargs...
    )
    for (i, (row, β)) in enumerate(zip(eachrow(xs), schedules[1:subsample_chains:end, end]))
        d = kde_lscv(row)
        StatsPlots.plot!(d, label=@sprintf("β = %.3f", β), c=pal[i])
        StatsPlots.scatter!(row, -i * ones(length(row)) / 50, c=pal[i], label=false, marker=:vline)
    end
    return p
end

function cumulative_round_trips_by_iter(runs, colors)
    p = StatsPlots.plot(title="Cumulative round trips vs iterations", xlabel="Iteration", legend=:outertopright)
    for (name, run) in runs
        iters = round_trip_completion_iters(run.index_process)
        StatsPlots.plot!(p, iters, 1:length(iters), label="$name", color=colors[name], seriestype=:steppost)
    end
    return p
end

function compare_cumulative_barriers(runs, colors; ind=nothing)
    p1 = StatsPlots.plot(title="Final barrier comparison", legend=:outertopright, size=(1200, 800))
    βs = 0.0:0.0001:1.0
    for (name, run) in runs
        barriers  = get_barriers(run.schedule_recorder)
        schedules = get_schedules(run.schedule_recorder)
        if ind === nothing
            StatsPlots.plot!(p1, βs, x -> barriers[end](x), label="$name", color=colors[name])
            StatsPlots.scatter!(p1, schedules[:, end], x -> barriers[end](x), label="$name", color=colors[name])
        else
            StatsPlots.plot!(p1, βs, x -> barriers[ind](x), label="$name", color=colors[name])
            StatsPlots.scatter!(p1, schedules[:, ind], x -> barriers[ind](x), label="$name", color=colors[name])
        end
    end
    return p1
end

function compare_rt_barrier_cumulative(runs)
    vruns = collect(runs)
    rts      = [round_trip_rate(run.index_process) for (_, run) in vruns]
    rt_ests1 = [1 / (1 + 2mean(get_Λ_rej(run.schedule_recorder))) for (_, run) in vruns]
    rt_ests2 = [1 / (1 + 2mean(get_Λ_acc(run.schedule_recorder))) for (_, run) in vruns]
    data = hcat(rts, rt_ests1, rt_ests2)
    labels = [n  for n in [L"\tau", L"(2\Lambda^{\mathrm{rej}} + 1)^{-1}", L"(2\Lambda^{\mathrm{acc}} + 1)^{-1}"] for _ in vruns ]
    names = [name  for _ in 1:3 for (name, _) in vruns]
    return groupedbar(
        names,
        data,
        group=labels,
        bar_position = :dodge,
        bar_width = 0.7,
        xrotation=20,
        title="Round trip and estimator comparison, cumulative",
        size=(1200, 800)
    )
end

function compare_rt_barrier_final_n(runs, n)
    vruns = collect(runs)
    rts      = [round_trip_rate(run.index_process.proc[:, end-(n * nsteps):end]) for (_, run) in vruns]
    rt_ests1 = [1 / (1 + 2mean(get_Λ_rej(run.schedule_recorder)[end-n:end])) for (_, run) in vruns]
    rt_ests2 = [1 / (1 + 2mean(get_Λ_acc(run.schedule_recorder)[end-n:end])) for (_, run) in vruns]
    data = hcat(rts, rt_ests1, rt_ests2)
    labels = [n  for n in [L"\tau", L"(2\Lambda^{\mathrm{rej}} + 1)^{-1}", L"(2\Lambda^{\mathrm{acc}} + 1)^{-1}"] for _ in vruns ]
    names = [name  for _ in 1:3 for (name, _) in vruns]
    return groupedbar(
        names,
        data,
        group=labels,
        bar_position = :dodge,
        bar_width = 0.7,
        title="Round trip and estimator comparison, final $n rounds",
        xrotation=20,
        size=(1200, 800)
    )
end

function compare_min_eigvals(runs, colors)
    p = StatsPlots.plot(title="Hessian min eigenvalue (Newton-TR)", xlabel="Newton step", legend=:outertopright)
    hline!([0.0], color=:black, linestyle=:dash, label=false)
    for (name, run) in runs
        if hasproperty(run.opt_state, :min_eigvals) && !isempty(run.opt_state.min_eigvals)
            StatsPlots.plot!(p, run.opt_state.min_eigvals, label=name, color=get(colors, name, :auto))
        end
    end
    return p
end

function compare_params(runs, colors)
    p1 = StatsPlots.plot(title="Parameter evolution", ylabel=L"\theta", legend=:outertopright)
    p2 = StatsPlots.plot(title="Step size evolution", ylabel="η", yscale=:log10, legend=:outertopright)
    for (name, run) in runs
        if !(occursin("no_opt", name))
            if isa(run.opt_state, TrustRegionState)
                opt = run.opt_state.inner_opt
            else
                opt = run.opt_state
            end
            StatsPlots.plot!(p1, stack(opt.xs; dims=1), color=colors[name], label=false)
            StatsPlots.plot!(p1, [NaN], color=colors[name], label="$name")
            if hasproperty(opt, :etas)
                StatsPlots.plot!(p2, max.(1e-20, opt.etas), label="$name", color=colors[name])
            end
        end
    end
    return StatsPlots.plot(p1, p2, layout=@layout([a; b]), size=(1200, 800))
end

_get_dim(x::AbstractVector, d::Int) = x[d]
_get_dim(x::Real, ::Int) = x

# Apply GBM transform T: z → x when the problem is a GBMProblem; identity otherwise.
_to_original_space(::Any, z) = z
_to_original_space(sp::GBMProblem, z) = NRPT.T(sp.m, z)

"""
    save_marginal_density_plots(run, name, dir)

For each recorded round and each variable dimension d, plots marginal KDE
densities coloured by chain temperature and saves to
`dir/densities/dim-d/name/round-r.svg`.

Samples are transformed back to the original space via T when the underlying
problem is a GBMProblem.
"""
function save_marginal_density_plots(run, name, dir)
    n_chains = size(run.x.xs, 1)
    n_rounds = length(run.x.rounds) - 1
    pal      = palette(:bamako, n_chains)
    sp       = run.problem.problem

    first_x  = _to_original_space(sp, run.x.xs[1, end])
    println(first_x)
    n_dims   = length(first_x)

    for d in 1:n_dims
        dim_dir = joinpath(dir, "densities", "dim-$d", name)
        mkpath(dim_dir)

        for r in 3:12:n_rounds
            samples = stack(_to_original_space.(Ref(sp), get_round_samples(run.x, r)))
            p = StatsPlots.plot(
                title="$name | dim $d | round $r",
                xlabel="x[$d]",
                legend=false,
                size=(800, 400)
            )
            for i in 1:n_chains
                vals = samples[d, i, :]
                de = kde_lscv(vals)
                StatsPlots.plot!(de, c=pal[i], alpha=0.6, linewidth=1.5)
            end
            Plots.savefig(p, joinpath(dim_dir, "round-$r.svg"))
        end
    end
end

function plot_all(
    runs::Dict,
    n_chains::Int,
    colors::Dict,
    dir::String;
    window_size::Int=10,
    subsample_chains::Int=1,
    loaded=false,
    show_exponents=true,
)
    runs = sort(runs)
    p = compare_barriers(runs, colors; window_size=1)
    Plots.savefig(p, joinpath(dir, "barriers.svg"))
    p = compare_barriers(runs, colors; window_size=window_size)
    Plots.savefig(p, joinpath(dir, "barriers-$window_size.svg"))
    if loaded
        p = compare_cumulative_barriers(runs, colors)
        Plots.savefig(p, joinpath(dir, "cumulative-barriers.svg"))
    else
        @info "Skipping plots of cumulative barriers, unavaliable in compresed file"
    end
    if show_exponents
        p = plot_exponents(runs, colors)
        Plots.savefig(p, joinpath(dir, "final-exponents.svg"))
        p = plot_exponents_log(runs, colors)
        Plots.savefig(p, joinpath(dir, "final-exponents-log.svg"))
    end
    p = compare_params(runs, colors)
    Plots.savefig(p, joinpath(dir, "param-evolution.svg"))
    p = compare_rt_barrier_cumulative(runs)
    Plots.savefig(p, joinpath(dir, "barrier-rt-comparison-cumulative.svg"))

    for (name, _) in runs
        p = plot_density(runs, name, n_chains; subsample_chains=subsample_chains)
        Plots.savefig(p, joinpath(dir, "final-density-$name.svg"))
    end
end
