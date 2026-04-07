# Unidentifiable product model: nf | nt, x, y ~ Binomial(nt, x·y), prior Uniform(0,1)².
# Posterior concentrates on the curve x·y = nf/nt in [0,1]², making it challenging to temper.

struct UnidProd <: SamplingProblem
    nt::Int
    nf::Int
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
    return V0(prob, x) + prob.nf * log(p) + (prob.nt - prob.nf) * log(1.0 - p)
end

function unidentifiable_product_slice_sampler(nt, nf; steps=5)
    return (
        UnidProd(nt, nf),
        IterExplorer(SliceSampler(), steps)
    )
end