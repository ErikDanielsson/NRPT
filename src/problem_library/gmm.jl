# Gaussian mixture model: equal-weight mixture of 2^d Gaussians at corners of {-mu, +mu}^d.
# Reference: isotropic Gaussian centred at 0.
function _gmm_corner_means(d, mu)
    [Float64[((k >> (j-1)) & 1) == 1 ? mu : -mu for j in 1:d] for k in 0:(2^d - 1)]
end

function gmm_slice_sampler(dimension; mu=3.0, sigma=0.3, sigma_ref=0.3, steps=5)
    D0    = MvNormal(zeros(dimension), sigma_ref^2 * I)
    means = _gmm_corner_means(dimension, mu)
    D1    = MixtureModel(MvNormal[MvNormal(μ, sigma^2 * I) for μ in means])
    return (
        GenericDistributionProblem(D0, D1),
        IterExplorer(SliceSampler(), steps)
    )
end
