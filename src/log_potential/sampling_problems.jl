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
struct GenericDistributionProblem <: DistributionProblem
    D0::Distribution
    D1::Distribution
end

sample_iid(problem::GenericDistributionProblem) = rand(problem.D0)

function V0(problem::GenericDistributionProblem, x)
    return -logpdf(problem.D0, x)
end

function V1(problem::GenericDistributionProblem, x)
    return -logpdf(problem.D1, x)
end

struct NormalProblem <: DistributionProblem
    μ0::Float64
    σ0::Float64
    μ1::Float64
    σ1::Float64
end

sample_iid(problem::NormalProblem) = rand(Normal(problem.μ0, problem.σ0))

function V0(problem::NormalProblem, x)
    return -logpdf(Normal(problem.μ0, problem.σ0), x)
end

function V1(problem::NormalProblem, x)
    return -logpdf(Normal(problem.μ1, problem.σ1), x)
end