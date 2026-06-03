using JuMP, Printf, StatsPlots, LaTeXStrings, Ipopt
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsfonts}")

paths = ["canonical", "q-path", "upper-trunc", "lower-trunc"]
path = paths[1]

if path == paths[1]
	@info "Using canonical path: π_β(θ) = π_0(θ) f(X | θ)^β"
	# This is the standard choice. q-parameter is ignored
	tempered_joint(p, f, β, q::Float64) = (p .* (f.^β))
	q = 0.0
elseif path == paths[2]
	@info "Using q-path: π_β,q(θ) = [π_0(θ)^(1 - q) + (1 - β) * (π_0(θ) f(X | θ))^(1 - q)]^(1 / (1 - q))"
	# The q parameter corresponds to a q-path. q = 0.0 corresponds to a mixture
	tempered_joint(p, f, β, q::Float64) =((1 - β) * p.^(1 - q) .+ β * (p .* f).^(1 - q)).^(1 / (1 - q))
	q = 0.1
elseif path == paths[3]
	@info "Using upper truncated path: π_β(θ) = π_0(θ) max(f(X | θ), 1 - β)" 
	# Somewhat similar to nested sampling -- temper the likelihood by cutting off the peaks
	_ϵ = 1e-6
	tempered_joint(p, f, β, q::Float64) = p .* max.(f, 1 - β)
	q = 0.0
elseif path == paths[4]
	@info "Using lower truncated path: π_β(θ) = π_0(θ) (min(f(X | θ), β) + (1 - β) q" 

	# Somewhat similar to nested sampling -- temper the likelihood by cutting off the peaks
	_ϵ = 1e-6
	tempered_joint(p, f, β::Float64, q::Float64) = p .* (min.(f, β) .+ (1 - β) * q)
	q = 0.0001
else 
	error("Unsupported path: $path")
end

function incorrect_coverage(α, n, m, β, ϵ, ϵ2, S, i)
	# p = ones(1, n) / n
    model = Model(Ipopt.Optimizer)
	set_attribute(model, "max_cpu_time", 60.0)
	# set_optimizer_attribute(model, "max_iter", 1000.)
	set_optimizer_attribute(model, "max_iter", 1000000)
    set_silent(model)
	# p = ones(1, n) / n
	# Ensure that each variable is greater than epsilon so that the support is the full set
    @variable(model, p[1:1, 1:n] >= ϵ2) 
    @variable(model, f[1:m, 1:n] >= ϵ2)
	
	# Minimize the total coverage over the (fixed) interval
    @objective(model, Min, sum(p[:, I]))
	
	# The prior should be a probability
    @constraint(model, sum(p) == 1)
	 # Likewise the likelihood for each parameter value
    @constraint(model, sum(f, dims=1) .== 1)
	# Impose the condition that the credible interval size should have atleast mass α
	@constraint(model,
				sum(S .* tempered_joint(p, f, β, q), dims=2) 
			.== (1 - α + ϵ2) * sum(tempered_joint(p, f, β, q), dims=2)
	)
	# But the the total mass under the posterior is strictly smaller than α
	@constraint(model, sum(S .* p) <= 1 - α - ϵ)
	
	for v in all_variables(model)
        set_start_value(v, rand())
	end
    optimize!(model)
	status = termination_status(model)
	assert_is_solved_and_feasible(model)
	return value(p), value(f)  
	# return p, value(f), β 
end

function hpd_from_pmf(p::AbstractVector, cred::Float64=0.95)
	x = collect(1:length(p))
    idx = sortperm(p, rev=true)
    cum = 0.0
    selected = Int[]
    for i in idx
        push!(selected, i)
        cum += p[i]
        if cum >= cred
            break
        end
    end
    selected_x = sort(x[selected])
	cover = sum(p[selected])
    return selected_x, cover
end

tempered_posterior(p, f, β, x=x) = tempered_joint(p, f, β, q)[x:x, :] ./ evidence(p, f, β, x)
evidence(p, f, β, x) = sum(tempered_joint(p, f, β, q), dims=2)[x:x, 1:1]

function contraint_qualification(p, f, β, α, ϵ, S, q)
    @info "Prior normalized: $(sum(p) .≈ 1.0 ? true : sum(p) .- 1)"
    @info "Likelihood normalized or diff: $(all(sum(f, dims=1) .≈ 1.0) ? true : sum(f, dims=1) .- 1)"
    @info "Credible interval sizes: $(sum(S .* tempered_joint(p, f, β, q), dims=2) ./ sum(tempered_joint(p, f, β, q), dims=2)) .>= $(1 - α)"
    @info "Coverage: $(sum(S .* p .* f)) < $(1 - α)"
end

function actual_coverage_plot(p, f)
    ηs = 0:0.001:1.0
    coverage = zeros(length(ηs))
    for (i, η) in enumerate(ηs)
        for x in 1:m
            inds = hpd_from_pmf(vec(tempered_posterior(p, f, η, x)), 1 - α)[1]
            posterior = tempered_posterior(p, f, 1.0, x)
            f_x = sum(evidence(p, f, 1.0, x))
			coverage[i] += sum(posterior[1, inds]) * f_x
		end
	end

    pal = palette(:mk_12, 10)
	p0 = plot(
        ηs, coverage,
        title=L"\textrm{Coverage}",
        xlabel=L"\beta",
        ylabel=L"\mathbb{P}[\theta \in C_\alpha^\beta(X)]",
        label=L"\mathbb{P}[\theta \in C_\alpha^\beta(X)]",
        # label="Actual coverage",
        legend=(-0.60, 1.00),
        left_margin = -25mm,
        ylims=(0-1e-2, 1+1e-2),
        linewidth=1; 
        palette=pal
    )
	hline!(p0, [1 - α], alpha=0.7, label=L"1 - \alpha = %$(1 - α)", linewidth=1, color=pal[2])
# 	vline!(p0, [β], color=:black, label=L"\textrm{Optimized } \beta", linewidth=1)
    styles = [:dash, :dot]
    p1 = plot(
        ηs, hcat([vcat([tempered_posterior(p, f, η, x) for η in ηs]...) for x in 1:m]...),
        title=L"\textrm{Tempered posteriors}",
        xlabel=L"\beta",
        ylabel=L"\pi_\beta(\theta \,\vert\, X)",
        label=reshape([L"\pi_\beta(\theta = %$i \,\vert\, x = %$j)"  for j in 1:m for i in 1:n], 1, m * n),
        linestyle=reshape([styles[j] for j in 1:m for i in 1:n], 1, m * n),
        legend=(1.05, 1.00),
        ylims=(0-1e-2, 1+1e-2),
        right_margin = 25mm,
        linewidth=1; 
        palette=pal
    )
	hline!(p1, [1 - α], alpha=0.7, label=L"1 - \alpha = %$(1 - α)", linewidth=1, color=pal[2])
    hline!
	p2 = bar(
        p',
        label="Prior",
        xlabel=L"\theta",
        ylabel=L"\pi(\theta)",
        title="Prior distribution solution",
        ylims=(0, 1),
        legend = :outertopright,
        legendfontsize=5,
        titlefontsize=11; palette=pal
    )
	p3 = groupedbar(f',
        label=reshape([L"\textrm{Likelihood } f(X = %$i \,\vert\, \theta)" for i in 1:m],
        (1, m)),
        xlabel="θ",
        title=L"\textrm{Likelihood sliced along } \theta",
        ylabel=L"f(\theta \,\vert\, X)",
        ylims=(0, 1),
        legend = :outertopright,
        legendfontsize=5,
        titlefontsize=11;
        palette=pal
    )
    l = @layout([a{0.25w} b{0.45w} [c; d]])
    empty = plot(
        framestyle = :none,
        grid = false,
        axis = nothing,
        legend = false,
        background_color = :transparent,
        foreground_color = :transparent
    )
    # l = @layout [a b]
    sp1 = plot(
        p0, p1 , p2, p3,
        layout = l,
        size=(900, 300),
        plot_title="Under-calibration in a categorical model",
        plot_titlevspan=0.1
    )
    sp1
    # l2 = @layout [a; b]
    # sp2 = plot(
    #     p2, p3,
    #     layout = l2,
    #     size=(300, 300),
    #     plot_title="Under-calibration in a categorical model",
    #     plot_titlevspan=0.1
    # )
    # return (sp1, sp2)
end

I = 1:1
α = 0.10
n = 3
m = 2
S = reshape([i ∈ I ? 1.0 : 0.0 for i ∈ 1:n], 1, n)
β = 0.80
ϵ = 1e-2
ϵ2 = 1e-4

function run()
    p, f = incorrect_coverage(α, n, m, β, ϵ, ϵ2, S, 100)
    contraint_qualification(p, f, β, α, ϵ, S, q)
    return p, f
end

p, f = run()
pl = actual_coverage_plot(p, f)
savefig(pl, "categorical-counterexample.pdf")