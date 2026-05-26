using Distributions, NRPT, Random, LinearAlgebra, StaticArrays
using DifferentiationInterface, ForwardDiff

# --- Model dimensions ---

const D = 1    

# --- Prior hyperparameters ---
const MU_PRIOR_MEAN = 0.0
const Y_OBS = 5

const MU_PRIOR_STD    = 0.1   
const OBS_STD  = 0.1   

const μ_vec = [MU_PRIOR_MEAN for i in 1:D]
const σ_vec = [MU_PRIOR_STD for i in 1:D]
const L = diagm(σ_vec)
const Σ = L^2

const D0 = MvNormal(μ_vec, Σ)

struct NormalModel <: NRPT.SamplingProblem end

function NRPT.V0(::NormalModel, params)
    return -(MU_PRIOR_MEAN - params[1])^2 / (2 * MU_PRIOR_STD^2)
end

NRPT.V1(m::NormalModel, params) = NRPT.V0(m, params) + normal_loglik(params)

function NRPT.sample_iid(::NormalModel)
    rand(D0)
end

function NRPT.sample_iid!(::NormalModel, x)
    rand!(D0, x)
end

function normal_loglik(params)
    return -(params[1] - Y_OBS)^2 / (2 * OBS_STD^2) 
end

struct NormalLikelihood <: NRPT.Likelihood end
const normal_gbm_prior = GaussianGBM(μ_vec, L)

NRPT.loglik(::NormalLikelihood, x) = normal_loglik(x)