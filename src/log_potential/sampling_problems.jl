
struct PosteriorProblem{T, S, W, D} <: SamplingProblem
    V0::T
    sample_iid::S
    V::W
    data::Vector{D}
end

sample_iid(problem::PosteriorProblem) = problem.sample_iid()

function V0(problem::PosteriorProblem, x)
    return problem.V0(x)
end

function V1(problem::PosteriorProblem, x)
    return problem.V0(x) + sum(problem.V(x, d) for d in problem.data)
end

abstract type DistributionProblem <: SamplingProblem end


struct MvUnivariate{D} <: Distributions.ContinuousMultivariateDistribution
    d::D
end

function Distributions._rand!(d::MvUnivariate, arr::AbstractArray{<:Real}) 
    arr[1] = rand(d.d)
    return arr
end

function Distributions._rand!(rng::AbstractRNG, d::MvUnivariate, arr::AbstractArray{<:Real}) 
    arr[1] = rand(rng, d.d)
    return arr
end

function Distributions._logpdf(d::MvUnivariate, arr)
    return logpdf(d.d, arr[1])
end

Distributions.length(d::MvUnivariate) = 1

struct GenericDistributionProblem <: DistributionProblem
    D0::MultivariateDistribution
    D1::MultivariateDistribution
end

sample_iid(problem::GenericDistributionProblem) = rand(problem.D0)
sample_iid!(problem::GenericDistributionProblem, x) = rand!(problem.D0, x)

function V0(problem::GenericDistributionProblem, x)
    return logpdf(problem.D0, x)
end

function V1(problem::GenericDistributionProblem, x)
    return logpdf(problem.D1, x)
end

struct NormalProblem <: DistributionProblem
    μ0::Float64
    σ0::Float64
    μ1::Float64
    σ1::Float64
end

const _LOG_SQRT_2PI = 0.9189385332046728  # log(sqrt(2π))

NormalProblem(D0::Normal, D1::Normal) = NormalProblem(params(D0)..., params(D1)...)

sample_iid(problem::NormalProblem) = randn() * problem.σ0 + problem.μ0
sample_iid(problem::NormalProblem) = randn() * problem.σ0 + problem.μ0

function V0(problem::NormalProblem, x)
    z = (x - problem.μ0) / problem.σ0
    return -0.5 * z * z - log(problem.σ0) - _LOG_SQRT_2PI
end

function V1(problem::NormalProblem, x)
    z = (x - problem.μ1) / problem.σ1
    return -0.5 * z * z - log(problem.σ1) - _LOG_SQRT_2PI
end

function exponents_to_params(problem::NormalProblem, η0, η1)
    σ2_β = (η0 / problem.σ0^2 + η1 / problem.σ1^2)^(-1)
    μ_β = σ2_β * (η0 * problem.μ0 / problem.σ0^2 + η1 * problem.μ1 / problem.σ1^2)
    return [μ_β, sqrt(σ2_β)]
end