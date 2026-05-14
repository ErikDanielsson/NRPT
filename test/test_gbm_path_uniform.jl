using NRPT, Test, DifferentiationInterface, LinearAlgebra, StatsPlots

# ── Minimal concrete Likelihood for testing ────────────────────────────────────
struct BinomialLikelihood <: Likelihood
    n::Float64
    s::Float64
end
NRPT.loglik(l::BinomialLikelihood, p) = l.s * log(prod(p)) + (l.n - l.s) * log(1 - prod(p))

# ── Helpers ───────────────────────────────────────────────────────────────────
make_schedule(n) = collect(range(0.0, 1.0, n))
make_x0(n, dim) = [randn(dim) for _ in 1:n]

@testset "ScalingGBMPath – StaticPath wrapping" begin
    path = ScalingGBMPath(2, LinearPath(), AutoForwardDiff())

    @test NRPT.extract_param(path) isa Vector{Float64}
    @test all(NRPT.extract_param(path) .≈ [0.0])

    NRPT.set_param!(path, [3.0])
    @test all(NRPT.extract_param(path) .≈ [3.0])

    lps = [1.0, 2.0]
    @test NRPT.log_potential(path, lps, 0.0) ≈ lps[1]
    @test NRPT.log_potential(path, lps, 1.0) ≈ lps[2]

    g = NRPT.gradient(path, lps, 0.5)
    @test g isa Vector{Float64}
end

# @testset "ScalingGBMPath – ParametrizedPath wrapping" begin
#     inner = QPath(0.5, AutoForwardDiff())
#     path  = ScalingGBMPath(2.0, inner, AutoForwardDiff())

#     t = NRPT.extract_param(path)
#     @test t isa AbstractVector
#     @test length(t) == 2
#     @test t[1] ≈ 2.0

#     NRPT.set_param!(path, [5.0, 0.0])
#     @test path.c ≈ 5.0
#     @test NRPT.extract_param(path.path) ≈ 0.0

#     lps = [1.0, 2.0]
#     @test NRPT.log_potential(path, lps, 0.0) ≈ lps[1]
#     @test NRPT.log_potential(path, lps, 1.0) ≈ lps[2]

#     g = NRPT.gradient(path, lps, 0.5)
#     @test g isa AbstractVector
#     @test length(g) == 2
# end

@testset "GBMProblem end-to-end" begin
    dim = 2
    gbm = UniformGBM(dim)
    lik = BinomialLikelihood(1000, 500)
    sp = GBMProblem(gbm, lik)

    z = randn(dim)
    @test (NRPT.V0(sp, z) ≈ -0.5sum(abs2, z))
    x = NRPT.T(gbm, z)
    @test NRPT.V1(sp, z) ≈ NRPT.V0(sp, z) + loglik(lik, x)
    @test length(NRPT.sample_iid(sp)) == dim

    n_chains = 10
    n_rounds = 12
    pal = palette(:RdBu, n_chains)
    path = ScalingGBMPath(3, LinearPath(), AutoForwardDiff())
    problem = PathProblem(sp, path, IterExplorer(SliceSampler(), 3))
    result = optimized_nrpt(
        make_x0(n_chains, dim), make_schedule(n_chains), problem,
        NoOptState(), 1;
        n_rounds = n_rounds, steps_per_round = n -> 2^n, record_samples = true
    )
    barrier = result.barriers[end](1.0)
    @test barrier > 0
    p = density(NRPT.T(gbm, stack(get_round_samples(result.x, n_rounds * 2))[1, :, :]'), title = "Barrier: $barrier", palette = pal)
    savefig(p, "uniform_density_linear.png")

    path = ScalingGBMPath(3, LinearPath(), AutoForwardDiff())
    NRPT.set_param!(path, [10.0, 10.0])
    problem = PathProblem(sp, path, IterExplorer(SliceSampler(), 3))
    result = optimized_nrpt(
        make_x0(n_chains, dim), make_schedule(n_chains), problem,
        NoOptState(), 1;
        n_rounds = n_rounds, steps_per_round = n -> 2^n, record_samples = true
    )
    barrier = result.barriers[end](1.0)
    @test barrier > 0
    p = density(NRPT.T(gbm, stack(get_round_samples(result.x, n_rounds * 2))[1, :, :]'), title = "Barrier: $barrier", palette = pal)
    savefig(p, "uniform_density_non_linear.png")

end
