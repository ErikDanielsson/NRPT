using NRPT, Printf, LaTeXStrings, KernelDensity
import CairoMakie

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

function compare_barriers(
    title, runs, colors, name_map;
    sch_tick = 1,
    opt_tick = 1,
    all_tick = 1,
    opt_onset = 7,
)
    f = CairoMakie.Figure()
    println("hela")
    fst = first(values(runs))
    n_sch_rounds = length(fst.schedule_recorder.Λ_rej)
    n_opt_rounds = length(get_Λ_opt_round(fst.loss_recorder))
    all_rounds = n_sch_rounds + n_opt_rounds

    p1 = CairoMakie.Axis(
        f[1, 2],
        title=L"$\Lambda$ (schedule rounds)",
        ylabel=L"$\Lambda$",
        xlabel="Schedule round",
        xticks=[1:sch_tick:(n_sch_rounds-sch_tick); n_sch_rounds]
    )
    p2 = CairoMakie.Axis(
        f[2, 1],
        title=L"$\overline{\mathrm{SKL}}$ (opt. rounds)",
        ylabel=L"$\overline{\mathrm{SKL}}$",
        xlabel="Opt. round",
        yscale=log10,
        xticks=[1:opt_tick:(n_opt_rounds-opt_tick); n_opt_rounds]
    )
    p3 = CairoMakie.Axis(
        f[2, 2:3],
        title=L"$\hat\tau$ (all rounds)",
        xlabel="Total round",
        ylabel=L"$\hat\tau$",
        xticks=[1:all_tick:(all_rounds-all_tick); all_rounds]
    )
    p4 = CairoMakie.Axis(
        f[1, 1],
        title=L"$\Lambda$ (opt. rounds)",
        ylabel=L"$\Lambda$",
        xlabel="Opt. round",
        xticks=[1:opt_tick:(n_opt_rounds-opt_tick); n_opt_rounds]
    )
    CairoMakie.linkyaxes!(p1, p4)
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
        @info "Barriers $name"
        c = colors[name]
        Λ_schedule = run.schedule_recorder.Λ_rej
        CairoMakie.lines!(p1, Λ_schedule, label=name_map[name], color=c)
        obj = get_objective_vals(run.loss_recorder)
        if !isempty(obj)
            println(obj)
            println(max.(1e-8, obj))
            CairoMakie.lines!(p2, max.(1e-8, obj), label=name_map[name], color=c)
        end
        rts   = count_round_trips_per_round(run.index_process)
        rates = rts ./ run.index_process.rounds
        CairoMakie.lines!(p3, 1:all_rounds, rates[2:end], label=name_map[name], color=c)
        λs = get_Λ_opt_round(run.loss_recorder)
        if !isempty(λs)
            CairoMakie.lines!(p4, λs, label=name_map[name], color=c)
        end
    end

    CairoMakie.vlines!(p1, [opt_onset], label="Opt. onset", color=:green, linestyle=:dash)
    CairoMakie.vlines!(p3, [opt_onset], label="Opt. onset", color=:green, linestyle=:dash)
    f[1, 3] = CairoMakie.Legend(f, p1, "Paths")

    f[0, :] = CairoMakie.Label(f, title)
    return f
end

function cumulative_round_trips(runs, colors, name_map)
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1],title="Cumulative round trips")
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
        c = colors[name]
        rts = cumsum(count_round_trips_per_round(run.index_process))
        CairoMakie.lines!(p, rts, label=name_map[name], color=c)
    end
    f[1, 2] = CairoMakie.Legend(f, p, "Paths")
    return f
end

function plot_exponents(runs, colors, name_map; ind=nothing)
    βs = 0:0.01:1
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1],)
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
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
        CairoMakie.lines!(e1s, e2s, label=name_map[name], color=colors[name])
    end
    f[1, 2] = CairoMakie.Legend(f, p, "Paths")
    p
end

function plot_exponents_log(runs, colors, name_map; ind=nothing, ϵ=1e-5)
    βs = ϵ:0.01:1.0 - ϵ
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1])
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
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
        CairoMakie.lines!(e1s, e2s, label=name_map[name], color=colors[name], yscale=:log10, xscale=:log10)
    end
    f[1, 2] = CairoMakie.Legend(f, p1, "Paths")
    p
end

function plot_schedule(runs, name; kwargs...)
    run = runs[name]
    schedules = get_schedules(run.schedule_recorder)
    n_chains = length(schedules[:, 1])
    pal = palette(:bamako, n_chains - 1)
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1],
        palette=pal,
        linewidth=2,
        title="Schedule evolution - $name",
        kwargs...
    )
    CairoMakie.lines!(p, schedules[2:end, :]', yscale=:log10)
    f[1, 2] = CairoMakie.Legend(f, p1, "Paths")
    return f
end

function whisker_plot(title, runs, name, n_chains, name_map, xlabel; round=nothing, subsample_chains=1, dim=nothing, kwargs...)
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1],
        title="Path samples $(name_map[name])",
        ylabel=L"\beta",
        xlabel=xlabel;
        kwargs...
    )
    run = runs[name]
    n_rounds = length(run.x.rounds) - 1
    schedules = get_schedules(run.schedule_recorder)
    r = isnothing(round) ? n_rounds : round
    xs = get_round_samples(run.x, r)[1:subsample_chains:end, :]
    if eltype(xs) <: AbstractVector
        d = isnothing(dim) ? 1 : dim
        xs = map(x -> x[d], xs)
    end
    pal = cgrad(:bamako)
    for (i, (row, β)) in enumerate(zip(eachrow(xs), schedules[1:subsample_chains:end, end]))
        CairoMakie.scatter!(p, row, β * ones(length(row)), color=pal[β], marker=:vline)
    end
    # CairoMakie.Colorbar(f[1, 2], limits = (0, 1), colormap = :bamako, flipaxis=true, ticks=[0., 1.], label=L"\beta")
    return f
end

function plot_density_and_whiskers(runs, name, n_chains, name_map; round=nothing, subsample_chains=1, dim=nothing, kwargs...)
    run = runs[name]
    pal = cgrad(:bamako)
    n_rounds = length(run.x.rounds) - 1
    r = isnothing(round) ? n_rounds : round
    xs = get_round_samples(run.x, r)[1:subsample_chains:end, :]
    if eltype(xs) <: AbstractVector
        d = isnothing(dim) ? 1 : dim
        xs = map(x -> x[d], xs)
    end
    schedules = get_schedules(run.schedule_recorder)
    f = CairoMakie.Figure()
    p1 = CairoMakie.Axis(f[1, 1],
        title="Path densities $(name_map[name]) (round $r)",
        kwargs...
    )
    p2 = CairoMakie.Axis(f[2, 1],
        title="Path samples $(name_map[name]) (round $r)",
        kwargs...
    )
    for (i, (row, β)) in enumerate(zip(eachrow(xs), schedules[1:subsample_chains:end, end]))
        d = kde_lscv(row)
        CairoMakie.lines!(p1, d, label=@sprintf("β = %.3f", β), color=pal[β])
        CairoMakie.scatter!(p2, row, ones(length(row)) , color=pal[i], marker=:vline)
    end
    CairoMakie.Colorbar(f[1, 2], limits = (0, 1), colormap = :bamako, flipaxis=true, ticks=[0., 1.], label=L"\beta")
    # f[1, 2] = CairoMakie.Legend(f, p, "Paths")
    return f
end

function plot_density(title, runs, name, n_chains, name_map, xlabel; round=nothing, subsample_chains=1, dim=nothing, kwargs...)
    run = runs[name]
    pal = cgrad(:bamako)
    n_rounds = length(run.x.rounds) - 1
    r = isnothing(round) ? n_rounds : round
    xs = get_round_samples(run.x, r)[1:subsample_chains:end, :]
    if eltype(xs) <: AbstractVector
        d = isnothing(dim) ? 1 : dim
        xs = map(x -> x[d], xs)
    end
    schedules = get_schedules(run.schedule_recorder)
    f = CairoMakie.Figure()
    p1 = CairoMakie.Axis(f[1, 1],
        title="$title - $(name_map[name])",
        ylabel="Density",
        xlabel=xlabel,
        kwargs...
    )
    for (i, (row, β)) in enumerate(zip(eachrow(xs), schedules[1:subsample_chains:end, end]))
        d = kde_lscv(row)
        CairoMakie.lines!(p1, d, label=@sprintf("β = %.3f", β), color=pal[β])
    end
    CairoMakie.Colorbar(f[1, 2], limits = (0, 1), colormap = :bamako, flipaxis=true, ticks=[0., 1.], label=L"\beta")
    # f[1, 2] = CairoMakie.Legend(f, p, "Paths")
    return f
end

function cumulative_round_trips_by_iter(title, runs, colors, name_map; barrier=nothing, ineff=nothing)
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1], title=title, ylabel="Cumulative round trips", xlabel="Iteration")

    # Plot each run
    max_iter = 1
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
        iters = round_trip_completion_iters(run.index_process)
        max_iter = max(max_iter, maximum(iters; init=0))
        CairoMakie.lines!(p, iters, 1:length(iters), label=name_map[name], color=colors[name])
    end

    if !isa(barrier, Nothing)
        # Plot the linear barrier on round trips
        CairoMakie.lines!(p, 1:max_iter, x -> x / (2 + 2barrier), label=L"$n \cdot \tau_\infty$ (linear path barrier)", linestyle=:dot, color=:grey)
    end

    if !isa(ineff, Nothing)
        # Plot the linear barrier on round trips
        CairoMakie.lines!(p, 1:max_iter, x -> x / (2 + 2ineff), label=L"$n \cdot\tau$ (linear schedule ineff.)", linestyle=:dash, color=:grey)
    end
    CairoMakie.axislegend(position=:lt) 
    # f[1, 2] = CairoMakie.Legend(f, p, "Paths")
    return f
end

function compare_cumulative_barriers(title, runs, colors, name_map; ind=nothing)
    f = CairoMakie.Figure()
    p1 = CairoMakie.Axis(f[1, 1], title=title, ylabel=L"$\Lambda(\beta)$ (final round)", xlabel=L"\beta")
    βs = 0.0:0.0001:1.0
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
        barriers  = get_barriers(run.schedule_recorder)
        schedules = get_schedules(run.schedule_recorder)
        if ind === nothing
            CairoMakie.lines!(p1, βs, x -> barriers[end](x), label=name_map[name], color=colors[name])
            CairoMakie.scatter!(p1, schedules[:, end], x -> barriers[end](x), color=colors[name])
        else
            CairoMakie.lines!(p1, βs, x -> barriers[ind](x), label=name_map[name], color=colors[name])
            CairoMakie.scatter!(p1, schedules[:, ind], x -> barriers[ind](x), color=colors[name])
        end
    end
    f[1, 2] = CairoMakie.Legend(f, p1, "Paths")
    return f
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

function compare_min_eigvals(runs, colors, name_map)
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1],title="Hessian min eigenvalue (Newton-TR)", xlabel="Newton step")
    hline!([0.0], color=:black, linestyle=:dash, label=false)
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
        if hasproperty(run.opt_state, :min_eigvals) && !isempty(run.opt_state.min_eigvals)
            CairoMakie.lines!(p, run.opt_state.min_eigvals, label=name, color=get(colors, name, :auto))
        end
    end
    f[1, 2] = CairoMakie.Legend(f, p1, "Paths")
    return f
end

function compare_params(title, runs, colors, name_map)
    f = CairoMakie.Figure()
    p1 = CairoMakie.Axis(f[1, 1], title=title, ylabel=L"\theta", xlabel="Optimization round")
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
        if !isa(run.opt_state, NoOptState)
            if isa(run.opt_state, TrustRegionState)
                opt = run.opt_state.inner_opt
            else
                opt = run.opt_state
            end
            params = stack(opt.xs; dims=1)
            for s in eachcol(params)
                CairoMakie.lines!(p1, s, color=colors[name])
            end
            CairoMakie.lines!(p1, [NaN], color=colors[name], label=name_map[name])
            if hasproperty(opt, :etas)
                CairoMakie.lines!(p2, max.(1e-20, opt.etas), label=name_map[name], color=colors[name])
            end
        end
    end
    f[1, 2] = CairoMakie.Legend(f, p1, "Paths")
    # axislegend(p1, position = :lb)
    return f
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
            f = Figure()
            p = CairoMakie.plot(
                title="$name | dim $d | round $r",
                xlabel="x[$d]",
            )
            for i in 1:n_chains
                vals = samples[d, i, :]
                de = kde_lscv(vals)
                CairoMakie.lines!(de, c=pal[i], alpha=0.6, linewidth=1.5)
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
    p = compare_barriers(runs, colors, name_map; window_size=1)
    Plots.savefig(p, joinpath(dir, "barriers.svg"))
    p = compare_barriers(runs, colors, name_map; window_size=window_size)
    Plots.savefig(p, joinpath(dir, "barriers-$window_size.svg"))
    if loaded
        p = compare_cumulative_barriers(runs, colors, name_map)
        Plots.savefig(p, joinpath(dir, "cumulative-barriers.svg"))
    else
        @info "Skipping plots of cumulative barriers, unavaliable in compresed file"
    end
    if show_exponents
        p = plot_exponents(runs, colors, name_map)
        Plots.savefig(p, joinpath(dir, "final-exponents.svg"))
        p = plot_exponents_log(runs, colors, name_map)
        Plots.savefig(p, joinpath(dir, "final-exponents-log.svg"))
    end
    p = compare_params(runs, colors, name_map)
    Plots.savefig(p, joinpath(dir, "param-evolution.svg"))
    p = compare_rt_barrier_cumulative(runs)
    Plots.savefig(p, joinpath(dir, "barrier-rt-comparison-cumulative.svg"))

    for (name, _) in sort(collect(pairs(runs)), by=x -> x[1])
        p = plot_density(runs, name, n_chains; subsample_chains=subsample_chains)
        Plots.savefig(p, joinpath(dir, "final-density-$name.svg"))
    end
end

function show_τ_paths(title, runs, colors, name_map; yscale=identity, ylims=(0, 1))
    f = CairoMakie.Figure()
    p = CairoMakie.Axis(f[1, 1], title=title, yscale=yscale, xlabel=L"\beta", ylabel=L"\tau(\beta)")
    if !isa(ylims, Nothing)
        CairoMakie.ylims!(p, ylims[1], ylims[2])
    end
    for (name, run) in sort(collect(pairs(runs)), by=x -> x[1])
        path = run.problem.path
        if isa(path, ScalingGBMPath)
            βs, raw_evals = NRPT.eval_schedule_basis(path.basis, path.c)
            evals = exp.(raw_evals)
            CairoMakie.lines!(p, βs, evals, label=name_map[name], color=colors[name])
        end
    end
    # Show the fixed endpoint
    CairoMakie.scatter!(p, [1], [1], color=:black)
    f[1, 2] = CairoMakie.Legend(f, p, "Paths")
    return f
end

function write_opt_params(runs, fn)
    opt_params = Dict()
    for (name, run) in runs
        opt_state = run.opt_state
        if !isa(opt_state, NoOptState)
            opt_params[name] = opt_state.xs
        end
    end
    jldsave(fn; opt_params)
    return
end