# MvNormal reference N(0, sigma0²·I) → target N(mu·1, sigma1²·I).
# Reference is fixed at 0 (translation invariance).
# Covers both symmetric benchmarks (sigma0=sigma1=1) and transport problems (sigma0≠sigma1 or mu large).
function mvnormal_slice_sampler(dimension; mu=1.0, sigma0=1.0, sigma1=1.0, steps=5)
    D0 = MvNormal(zeros(dimension), sigma0^2 * I)
    D1 = MvNormal(mu * ones(dimension), sigma1^2 * I)
    return (
        GenericDistributionProblem(D0, D1),
        IterExplorer(SliceSampler(), steps)
    )
end

# GBM version: reference N(0, I) in z-space, target N(mu·1, sigma1²·I).
# T(z) = z (identity), so the GBM reference and the sampling space coincide.
function normal_gbm(dimension; mu=5.0, sigma=0.1, steps=5)
    gbm = GaussianGBM(zeros(dimension), Matrix(1.0I, dimension, dimension))
    lik = GaussianLikelihood(mu * ones(dimension), sigma)
    sp  = GBMProblem(gbm, lik)
    return sp, IterExplorer(SliceSampler(), steps)
end
