using Random, Distributions, LinearAlgebra, ColorSchemes, Plots, LaTeXStrings
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsfonts}")

function sample_parameters(Λ, n, ϵ=0.01)
    p = size(Λ, 1) 
    μ0 = randn(p, 1)
    Σ0 = rand(Wishart(p + 1., Λ)) + ϵ * I
    Σ1 = rand(Wishart(p + 1., Λ)) + ϵ * I
    X = rand(MvNormal(vec(μ0), Σ1), n)
    return μ0, Σ0, Σ1, X
end


struct Ellipsoid{T}
    μ::Vector{T}
    coord_transform::Matrix{T} # The map from ellipsoid to sphere
    rev_coord_transform::Matrix{T} # The map from sphere to ellipsoid
    r::T # Radius of untransformed sphere
end

function compute_credible_ellipsoid(posterior::MvNormal, α::Float64)::Ellipsoid
    # Transform the distribution into a standard multivariate normal
    p = length(posterior.μ)
    MT = cholesky(posterior.Σ).L.data
    d = Chi(p)
    r = quantile(d, 1 - α)
    return Ellipsoid(posterior.μ, MT, inv(MT), r)
end

function sample_boundary(e::Ellipsoid{Float64}, N::Int, random::Bool=false)
    p = length(e.μ)

    if random
        grid = randn(p, N)
        grid ./= sqrt.(sum(grid.^2, dims=1))
        grid *= e.r
    else 
        grid = sphere_grid(p, N)
    end
    return e.μ .+ e.coord_transform * grid
end

function sphere_grid(n, k)
    # --- Special case: S¹ (circle) ---
    if n == 2
        θ = range(0, stop=2π, length=k)
        points = [[cos(t), sin(t)] for t in θ]
        return reduce(hcat, (p for p in points))
    end

    # --- General case: S^{n−1}, n ≥ 3 ---
    angles = [range(0, stop=π, length=k) for _ in 1:(n-2)]
    push!(angles, range(0, stop=2π, length=k))

    function to_cart(theta)
        x = zeros(n)
        prod = 1.0
        # first n−1 coordinates
        for i in 1:(n-1)
            if i < n-1
                x[i] = prod * sin(theta[i])
                prod *= sin(theta[i])
            else
                x[i] = prod * cos(theta[i])
            end
        end
        # last coordinate
        x[n] = prod
        return x
    end

    Θ = Iterators.product(angles...)
    points = [to_cart(collect(t)) for t in Θ]

    reduce(hcat, (p for p in points))
end


function indicator(e::Ellipsoid{Float64}, x::Vector{Float64})
    return sum((e.rev_coord_transform * (x - e.μ)).^2) ≤ e.r^2
end

function indicator(e::Ellipsoid{Float64}, x::Matrix{Float64})
    return sum((e.rev_coord_transform * (x .- e.μ)).^2, dims=1) .≤ e.r^2
end

function ellipsoid_plot()
    # Sample the prior and likelihood parameters
    μ0, Σ0, Σ1, X = sample_parameters([1. 0.; 0. 1.], 10)
    α = 0.90

    βs = 0:0.1:1

    tempered_posterior(β::Float64) = tempered_posterior(μ0, Σ0, Σ1, X, β)
    ellipsoid(dist::MvNormal) = compute_credible_ellipsoid(dist, α)

    posteriors = tempered_posterior.(βs)
    ellipsoids = ellipsoid.(posteriors)
    boundaries = sample_boundary.(ellipsoids, 1000)

    plot(aspect_ratio=:equal)
    for (β, E) in zip(βs, boundaries)
        plot!(E[1, :], E[2, :],  color=cgrad(:thermal, rev=true)[β], label="Temperature $β")
    end
    plot!(legend=false)
end


# 
# Semi-analytic computation of the coverage of the tempered posteriors
# 
function tempered_posterior(μ0, Σ0, Σ1, X, β)
    p, n = size(X)
    x_avg = sum(X; dims=2)
    A0 = inv(Σ0)
    b0 = A0 * μ0
    A1 = β * n * inv(Σ1)
    b1 = A1 * x_avg
    Σn = inv(A0 + A1)
    μn = Σn * (b0 + b1)
    return MvNormal(vec(μn), Σn)
end

function tempering_distortion(Σ0, Σ1, β, n)
    λs = eigvals(Σ1 * inv(Σ0))
    w = 1 .- n * β * (1 - β) ./ (λs .+ n * β)
    return w
end

function MC_coverage(Σ0, Σ1, β, m, N, α; seed=1)
    rng = Random.seed!(seed)
    n = size(Σ0, 1) 
    w = tempering_distortion(Σ0, Σ1, β, m)
    c_α = quantile(Chisq(n), 1 - α)
    sample = rand(rng, Chisq(1), N, n) * w
    return mean(sample .<= c_α)# , std(sample .<= c_α)
end

function coverage_plot(Σ0, Σ1, ms, α; N=100000, δ=0.01, β_min=0, β_max=1)
    βs = β_min:δ:β_max
    coverage(β, m) = MC_coverage(Σ0, Σ1, β, m, N, α; seed=1)# Random.random_seed())
    asymp_coverage(β) = MC_coverage(Σ0, zeros(size(Σ1)), β, 1, N, α; seed=1)# Random.random_seed())
    distortion(β, m) = tempering_distortion(Σ0, Σ1, β, m)
    n = size(Σ0, 1)
    # coverages = coverage.(βs)
    # mean_coverage = 
    pal = palette(:mk_12, 10)
	c_p = plot(
        βs, ones(length(βs)) * (1 - α),
        alpha=0.7,
        label="Prescribed coverage",
        linewidth=1;
        palette=pal)
    for m in ms
        c_p = plot!(
            c_p,
            βs, coverage.(βs, m),
            xlabel=L"\beta",
            title=L"Coverage ($\mathbb{P}[\theta \in C_\alpha^\beta(X)]$)",
            label=L"\textrm{Actual coverage } (m = %$m)",
            legend=(-1.0, 1.0)
        )
    end
    c_p = plot!(
        c_p,
        βs[2:end], asymp_coverage.(βs[2:end]),
        xlabel=L"\beta",
        title=L"Coverage ($\mathbb{P}[\theta \in C_\alpha^\beta(X)]$)",
        label="Asymptotic coverage",
        color=pal[4]
    )
    d_p = plot(
        βs, βs,
        xlabel=L"\beta",
        label = L"\beta",
        title=L"$\chi^2$-mixture weights ($w_i^\beta$)";
        color=pal[4]
    )
    for m in ms
        d_p = plot!(
            d_p,
            βs, hcat(distortion.(βs, m)...)',
            xlabel="β",
            label = reshape([L"w_{%$i} (m = %$m)" for i in 1:n], (1, n)),
            title=L"$\chi^2$-mixture weights ($w_i^\beta$)";
            palette=pal
        )
    end
    
    plot(
        c_p, d_p,
        layout=2,
        plot_title="Over-calibration in a conjugate Gaussian model",
        plot_titlevspan=0.1, 
        size=(600, 300)
    )
end

function generate_example(; seed=1, n=3, ms=[10, 100], α=0.05, N=1000000)
    rng = Random.seed!(seed)
    Σ0 = I[1:n, 1:n]
    Σ1 = diagm(1:3)#  rand(rng, Wishart(10, I[1:n, 1:n]))
    println("Σ0: $Σ0")
    println("Σ1: $Σ1")
    println("Matrix: $(inv(Σ0) * Σ1)")
    coverage_plot(Σ0, Σ1, ms, α; N=N)
end