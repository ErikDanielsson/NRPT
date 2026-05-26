using Distributions, NRPT, Random, LinearAlgebra, StaticArrays
import RDatasets
using DifferentiationInterface, ForwardDiff

# --- Model dimensions ---


const sdata = let 
    data = RDatasets.dataset("MASS", "galaxies").x1
    data[78] = 26960
    n_obs = length(data) 
    SVector{n_obs}(data / 1000.)
end

const N_OBS = length(sdata)

const K     = 5    # number of mixture components
const N_DIM = 3K   # μ (K) + log_σ (K) + w (K)

# --- Prior hyperparameters ---
const MU_PRIOR_MEAN = 0.0
const LOGΣ_PRIOR_MEAN = 0.0
const W_PRIOR_MEAN = 0.0

const μ_vec = vcat(
    [MU_PRIOR_MEAN for k in 1:K],
    [LOGΣ_PRIOR_MEAN for k in 1:K],
    [W_PRIOR_MEAN for k in 1:K]
)

const MU_PRIOR_STD    = 1.0   # Normal(0, MU_PRIOR_STD) on each μ_k
const LOGΣ_PRIOR_STD  = 1.0   # Normal(0, LOGΣ_PRIOR_STD) on each log_σ_k
const W_PRIOR_STD     = 1.0   # Normal(0, W_PRIOR_STD) on each unconstrained weight

const σ_vec = vcat(
    [MU_PRIOR_STD for k in 1:K],
    [LOGΣ_PRIOR_STD for k in 1:K],
    [W_PRIOR_STD for k in 1:K]
)
const L = diagm(σ_vec)
const Σ = L^2

const D0 = MvNormal(μ_vec, Σ)

#   params[1:K]        = μ      (component means)
#   params[K+1:2K]     = log_σ  (log component standard deviations, σ_k = exp(log_σ_k))
#   params[2K+1:3K]    = w      (unconstrained weights; mixing proportions = softmax(w))

struct MarginalMixtureModel <: NRPT.SamplingProblem end

function NRPT.V0(::MarginalMixtureModel, params)
    lp = 0.0
    @inbounds for k in 1:K
        lp += logpdf(Normal(MU_PRIOR_MEAN, MU_PRIOR_STD),   params[k])
        lp += logpdf(Normal(LOGΣ_PRIOR_MEAN, LOGΣ_PRIOR_STD), params[K+k])
        lp += logpdf(Normal(W_PRIOR_MEAN, W_PRIOR_STD),     params[2K+k])
    end
    return lp
end

NRPT.V1(m::MarginalMixtureModel, params) = NRPT.V0(m, params) + mixture_loglik(params)

function NRPT.sample_iid(::MarginalMixtureModel)
    rand(D0)
end

function NRPT.sample_iid!(::MarginalMixtureModel, x)
    rand!(D0, x)
end

@inline function logsumexp_buffer(buf)
    m = maximum(buf)
    s = 0.0
    @inbounds for v in buf
        s += exp(v - m)
    end
    return m + log(s)
end

function mixture_loglik(params)
    # Precompute per-component constants
    lse_w = logsumexp_buffer(@view params[2K+1:3K])

    lnorm    = MVector{K, Float64}(undef)   # -0.5*log(2π) - log_σ_k
    inv2σ2   = MVector{K, Float64}(undef)
    lπ       = MVector{K, Float64}(undef)
    @inbounds for k in 1:K
        lσk       = params[K+k]
        lnorm[k]  = -0.5 * log(2π) - lσk
        inv2σ2[k] = 0.5 * exp(-2lσk)
        lπ[k]     = params[2K+k] - lse_w
    end

    log_components = MVector{K, Float64}(undef)
    lsum = 0.0
    @inbounds for i in 1:N_OBS
        y = sdata[i]
        for k in 1:K
            log_components[k] = lπ[k] + lnorm[k] - inv2σ2[k] * (y - params[k])^2
        end
        lsum += logsumexp_buffer(log_components)
    end

    return isfinite(lsum) ? lsum : -Inf
end

struct MMM <: NRPT.Likelihood end
const mmm_gbm_prior = GaussianGBM(μ_vec, L)

NRPT.loglik(::MMM, x) = mixture_loglik(x)