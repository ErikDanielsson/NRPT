using NRPT, Test, DifferentiationInterface, LinearAlgebra, StatsPlots
using LogExpFunctions

# ── Minimal concrete Likelihood for testing ────────────────────────────────────
struct BimodalLikelihood <: Likelihood
    μ::Float64
    σ::Float64
end
NRPT.loglik(l::BimodalLikelihood, x) = logaddexp(
    -0.5 * sum(abs2, (x .+ l.μ) ./ l.σ),
    -0.5 * sum(abs2, (x .- l.μ) ./ l.σ)
)

# ── Helpers ───────────────────────────────────────────────────────────────────
make_schedule(n) = collect(range(0.0, 1.0, n))
make_x0(n, dim) = [randn(dim) for _ in 1:n]

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
    dim = 1
    gbm = GaussianGBM(zeros(dim), Matrix(1.0I, dim, dim))
    lik = BimodalLikelihood(5.0, 0.1)
    sp = GBMProblem(gbm, lik)

    z = randn(dim)
    @test (NRPT.V0(sp, z) ≈ -0.5sum(abs2, z))
    x = NRPT.T(gbm, z)
    @test NRPT.V1(sp, z) ≈ NRPT.V0(sp, z) + loglik(lik, x)
    @test length(NRPT.sample_iid(sp)) == dim

    n_chains = 20
    path = ScalingGBMPath(2, LinearPath(), AutoForwardDiff())
    problem = PathProblem(sp, path, IterExplorer(SliceSampler(), 5))
    result = optimized_nrpt(
        NRPTConfig(
            make_x0(n_chains, dim), problem;
            seed = 1, n_rounds = 10, steps_per_round = n -> 2^n, record_samples = true
        )
    )
    barrier = result.schedule_recorder.barriers[end](1.0)
    @test barrier > 0
    p = density(stack(result.x.xs)[1, :, :]', title = "Barrier: $barrier")
    savefig(p, "bimodal_density_linear.png")

    path = ScalingGBMPath(3, LinearPath(), AutoForwardDiff())
    NRPT.set_param!(path, [10000.0, 10000.0])
    problem = PathProblem(sp, path, IterExplorer(SliceSampler(), 3))
    result = optimized_nrpt(
        NRPTConfig(
            make_x0(n_chains, dim), problem;
            seed = 1, n_rounds = 10, steps_per_round = n -> 2^n, record_samples = true
        )
    )
    barrier = result.schedule_recorder.barriers[end](1.0)
    @test barrier > 0
    p = density(stack(result.x.xs)[1, :, :]', title = "Barrier: $barrier")
    savefig(p, "bimodal_density_non_linear.png")

end
