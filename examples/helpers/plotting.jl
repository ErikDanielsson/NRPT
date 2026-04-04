using NRPT, Plots, Printf, LaTeXStrings, KernelDensity

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

function compare_barriers(runs, n_rounds, colors; window_size=10)
    p1 = plot(title="Barrier comparison", yscale=:log, legend=:outertopright)
    p2 = plot(title="SKL comparison", yscale=:log, legend=:outertopright)
    p3 = plot(title="Round trips", legend=:outertopright)
    for (name, run) in runs
        c = colors[name]
        plot!(p1, moving_average([b(1.0) for b in run.barriers], window_size), label="$name", color=c)
        if length(run.objective_vals) > 0 && run.objective_vals[1] !== nothing
            println(name)
            plot!(p2, max.(1e-8, moving_average(run.objective_vals, window_size)), label="$name", color=c)
        end
        rts = count_round_trips_per_round(run.index_process, n_rounds)
        plot!(p3, moving_average(rts, window_size), label="$name", color=c)
    end
    return plot(p1, p2, p3, layout=@layout([a; b; c]), plot_title="Optimization comparison", size=(1200, 800))
end

function cumulative_round_trips(runs, n_rounds, colors)
    p = plot(title="Cumualtive round trips", legend=:outertopright)
    for (name, run) in runs
        c = colors[name]
        rts = cumsum(count_round_trips_per_round(run.index_process, n_rounds))
        plot!(p, rts, label="$name", color=c)
    end
    return p
end

function plot_exponents(runs, colors; ind=nothing)
    βs = 0:0.01:1
    p = plot()
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
        plot!(e1s, e2s, label="$name", color=colors[name]) 
    end
    p
end

function plot_exponents_log(runs, colors; ind=nothing, ϵ=1e-5)
    βs = ϵ:0.01:1.0 - ϵ
    p = plot(size=(1200, 800), legend=:outertopright)
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
        plot!(e1s, e2s, label="$name", color=colors[name], yscale=:log10, xscale=:log10) 
    end
    p
end

function plot_density(runs, name, samples, n_chains; final_ind=nothing, subsample_chains=1, dim=nothing,  kwargs...)
    run = runs[name]
    pal = palette(:bamako, n_chains)
    if final_ind == nothing
        xs = run.x[1:subsample_chains:end, end-samples:end]
    else
        xs = run.x[1:subsample_chains:end, final_ind-samples:final_ind]
    end
    if eltype(xs) <: AbstractVector
        d = isnothing(dim) ? 1 : dim
        xs = map(x -> x[d], xs)
    end
    n_shown = size(xs, 1)
    p = plot(
        palette=pal,
        linewidth=2,
        title="Density plot $name",
        legend=:outertopright,
        size=(1200, 800);
        kwargs...
    )
    for (i, (row, β)) in enumerate(zip(eachrow(xs), run.schedules[1:subsample_chains:end, end]))
        d = kde_lscv(row)
        plot!(d, label=@sprintf("β = %.3f", β), c=pal[i])
        scatter!(row, -i * ones(length(row)) / 50, c=pal[i], label=false, marker=:vline)
    end
    # p = plot(
    #     xs',
    #     palette=pal,
    #     linewidth=2,
    #     title="Density plot $name",
    #     label=reshape([@sprintf("β = %.3f", f) for f in run.schedules[1:subsample_chains:end, end]], 1, n_shown),
    #     legend=:outertopright,
    #     size=(1200, 800);
    #     kwargs...
    # )
    return p
end

function compare_cumulative_barriers(runs, colors; ind=nothing)
    p1 = plot(title="Final barrier comparison", legend=:outertopright, size=(1200, 800))
    βs = 0.0:0.0001:1.0
    for (name, run) in runs
        if ind === nothing
            plot!(p1, βs, x -> run.barriers[end](x), label="$name", color=colors[name])
            scatter!(p1, run.schedules[:, end], x -> run.barriers[end](x), label="$name", color=colors[name])
        else
            plot!(p1, βs, x -> run.barriers[ind](x), label="$name", color=colors[name])
            scatter!(p1, run.schedules[:, ind], x -> run.barriers[ind](x), label="$name", color=colors[name])
        end
    end
    return p1
end

function compare_rt_barrier_cumulative(runs)
    vruns = collect(runs)
    rts = [round_trip_rate(run.index_process) for (_, run) in vruns]
    rt_ests1 = [1 / (1 + 2mean(run.Λ_rej)) for (_, run) in vruns]
    rt_ests2 = [1 / (1 + 2mean(run.Λ_acc)) for (_, run) in vruns]
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

function compare_rt_barrier_final_n(runs, n, nsteps)
    vruns = collect(runs)
    rts = [round_trip_rate(run.index_process[:, end-(n * nsteps):end]) for (_, run) in vruns]
    rt_ests1 = [1 / (1 + 2mean(run.Λ_rej[end-n:end])) for (_, run) in vruns]
    rt_ests2 = [1 / (1 + 2mean(run.Λ_acc[end-n:end])) for (_, run) in vruns]
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
    p = plot(title="Hessian min eigenvalue (Newton-TR)", xlabel="Newton step", legend=:outertopright)
    hline!([0.0], color=:black, linestyle=:dash, label=false)
    for (name, run) in runs
        if hasproperty(run, :min_eigvals) && !isempty(run.min_eigvals)
            plot!(p, run.min_eigvals, label=name, color=get(colors, name, :auto))
        end
    end
    return p
end

function compare_params(runs, colors)
    p1 = plot(title="Parameter evolution", ylabel=L"\theta", legend=:outertopright)
    p2 = plot(title="Step size evolution", ylabel="η", yscale=:log10, legend=:outertopright)
    for (name, run) in runs
        if !(occursin("no_opt", name))
            if isa(run.opt_state, TrustRegionState)
                opt = run.opt_state.inner_opt
            else
                opt = run.opt_state
            end
            plot!(p1, stack(opt.xs; dims=1), color=colors[name], label=false)
            plot!(p1, [NaN], color=colors[name], label="$name")
            if hasproperty(opt, :etas)
                plot!(p2, max.(1e-20, opt.etas), label="$name", color=colors[name])
            end
        end
    end
    return plot(p1, p2, layout=@layout([a; b]), size=(1200, 800))
end

function plot_all(
    runs::Dict,
    n_rounds::Int,
    n_chains::Int,
    colors::Dict,
    round_size::Int,
    dir::String;
    window_size::Int=10,
    subsample_chains::Int=1,
    loaded=false,
    show_exponents=true,
)
    runs = sort(runs)
    p = compare_barriers(runs, n_rounds, colors; window_size=1)
    Plots.savefig(p, joinpath(dir, "barriers.svg"))
    p = compare_barriers(runs, n_rounds, colors; window_size=window_size)
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
    n = 10
    p = compare_rt_barrier_final_n(runs, n, round_size)
    Plots.savefig(p, joinpath(dir, "barrier-rt-comparison-last-$n.svg"))
    p = compare_rt_barrier_cumulative(runs)
    Plots.savefig(p, joinpath(dir, "barrier-rt-comparison-cumulative.svg"))

    for (name, _) in runs in 
        if !(startswith(name, "no-opt"))
        end
       p = plot_density(runs, name, round_size, n_chains; subsample_chains=subsample_chains)
        Plots.savefig(p, joinpath(dir, "final-density-$name.svg"))
    end 
end