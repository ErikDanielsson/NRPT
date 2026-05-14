# Gaussian mixture model: equal-weight mixture of 2^d Gaussians at corners of {-mu, +mu}^d.
# Reference: isotropic Gaussian centred at 0.
function _gmm_corner_means(d, mu)
    return [Float64[((k >> (j - 1)) & 1) == 1 ? mu : -mu for j in 1:d] for k in 0:(2^d - 1)]
end

function gmm_slice_sampler(dimension; mu = 3.0, sigma = 0.3, sigma_ref = 0.3, steps = 5)
    D0 = MvNormal(zeros(dimension), sigma_ref^2 * I)
    means = _gmm_corner_means(dimension, mu)
    D1 = MixtureModel(MvNormal[MvNormal(μ, sigma^2 * I) for μ in means])
    return (
        GenericDistributionProblem(D0, D1),
        IterExplorer(SliceSampler(), steps),
    )
end

# GBM version: reference N(0, I) in z-space, target is N(0,I) * GMM (product distribution).
# T(z) = z (identity), so z-space = x-space.
struct GMLikelihood{D <: MultivariateDistribution} <: Likelihood
    target::D
end

loglik(l::GMLikelihood, x) = logpdf(l.target, x)

function gmm_gbm(dimension; mu = 3.0, sigma = 0.3, steps = 5)
    means = _gmm_corner_means(dimension, mu)
    target = MixtureModel(MvNormal[MvNormal(μ, sigma^2 * I) for μ in means])
    gbm = GaussianGBM(zeros(dimension), Matrix(1.0I, dimension, dimension))
    sp = GBMProblem(gbm, GMLikelihood(target))
    return sp, IterExplorer(SliceSampler(), steps)
end
