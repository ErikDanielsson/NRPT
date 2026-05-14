# Unidentifiable product model: nf | nt, x, y ~ Binomial(nt, x·y), prior Uniform(0,1)².
# Posterior concentrates on the curve x·y = nf/nt in [0,1]², making it challenging to temper.

struct UnidProd <: SamplingProblem
    n::Int
    s::Int
end

function V0(::UnidProd, x)
    for i in eachindex(x)
        (x[i] <= 0.0 || x[i] >= 1.0) && return -Inf
    end
    return zero(eltype(x))
end

sample_iid(::UnidProd) = rand(2)

function V1(prob::UnidProd, x)
    p = clamp(x[1] * x[2], eps(), 1.0 - eps())
    return V0(prob, x) + prob.s * log(p) + (prob.n - prob.s) * log(1.0 - p)
end

function unidentifiable_product_slice_sampler(nt, nf)
    return (
        UnidProd(nt, nf),
        SliceSampler(),
    )
end

struct UnidProdLikelihood <: Likelihood
    n::Int
    s::Int
end

function loglik(l::UnidProdLikelihood, x)
    p = clamp(x[1] * x[2], eps(), 1.0 - eps())
    return l.s * log(p) + (l.n - l.s) * log(1.0 - p)
end

function unidentifiable_product_gbm(nt::Int, nf::Int)
    gbm = UniformGBM(2)
    lik = UnidProdLikelihood(nt, nf)
    sp = GBMProblem(gbm, lik)
    explorer = SliceSampler()
    return sp, explorer
end
