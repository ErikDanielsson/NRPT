# ── Abstract GBM interface ────────────────────────────────────────────────────
# A GBM (Gaussian Base Measure) model defines:
#   - a latent space z ~ N(0, I)   (the reference)
#   - a transformation T: z → x    (to the observable space)
#   - the reference potential V0(z) = ||z||²  (∝ -log N(0,I), up to constant)
#
# PathProblem pairs a GBMProblem (SamplingProblem) with a ScalingGBMPath.

abstract type GBM end

# Reference potential (negative log density of N(0,I), dropping constant).
V0(::GBM, z) = -0.5sum(abs2, z)

# Transformation z → x (must be overridden).
T(g::GBM, z) = throw(MethodError(T, (g, z)))

Base.length(g::GBM) = throw(MethodError(Base.length, (g,)))

sample_iid(g::GBM) = randn(length(g))
sample_iid!(::GBM, x) = randn!(x)

abstract type BaseMeasureChange end

V0β(bmc::BaseMeasureChange, gbm::GBM, z::AbstractVector) =
    throw(MethodError(V0β, (bmc, gbm, z)))
V0β(bmc::BaseMeasureChange, V0::Real) =
    throw(MethodError(V0β, (bmc, V0)))

abstract type Likelihood end

# Log-likelihood evaluated at x (must be overridden).
loglik(l::Likelihood, x) = throw(MethodError(loglik, (l, x)))

# ── GBMProblem: SamplingProblem backed by a GBM + Likelihood ─────────────────
struct GBMProblem{M <: GBM, L <: Likelihood} <: SamplingProblem
    m::M
    l::L
end

V0(p::GBMProblem, z) = V0(p.m, z)
V1(p::GBMProblem, z) = V0(p, z) + loglik(p.l, T(p.m, z))
sample_iid(p::GBMProblem) = sample_iid(p.m)
sample_iid!(p::GBMProblem, x) = sample_iid!(p.m, x)


"""
    GaussianGBM(μ, L)

Gaussian base measure with mean `μ` and Cholesky factor `L` (so Σ = L Lᵀ).
The transformation is T(z) = μ + L z.
"""
struct GaussianGBM{Tv <: AbstractVector{<:Real}, M <: AbstractMatrix{<:Real}} <: GBM
    μ::Tv
    L::M
end

T(g::GaussianGBM, z) = g.μ + g.L * z
Base.length(g::GaussianGBM) = length(g.μ)

"""
    UniformGBM(dim)

Uniform base measure on [0,1]^dim via the probit transformation T(z) = Φ(z).
"""
struct UniformGBM <: GBM
    dim::Int
end

T(::UniformGBM, z) = cdf.(Normal(), z)
Base.length(g::UniformGBM) = g.dim

"""
    BoundedUniformGBM(lb, ub)

Uniform base measure on [lb, ub] via the scaled probit transform T(z) = lb .+ (ub .- lb) .* Φ(z).
"""
struct BoundedUniformGBM <: GBM
    lb::Vector{Float64}
    ub::Vector{Float64}
end

T(g::BoundedUniformGBM, z) = g.lb .+ (g.ub .- g.lb) .* cdf.(Normal(), z)
Base.length(g::BoundedUniformGBM) = length(g.lb)

# ── Concrete Likelihood types ─────────────────────────────────────────────────

"""
    GaussianLikelihood(μ, σ)

Isotropic Gaussian likelihood: log p(x) ∝ -||x - μ||² / (2σ²).
"""
struct GaussianLikelihood{Tv} <: Likelihood
    μ::Tv
    σ::Float64
end

loglik(l::GaussianLikelihood, x) = -0.5 / l.σ^2 * sum(abs2, x .- l.μ)
