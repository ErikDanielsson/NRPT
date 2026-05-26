using Distributions, NRPT, Random, LinearAlgebra, DelimitedFiles
using LogExpFunctions

# --- Data loading  ---

const _raw = readdlm(joinpath(@__DIR__, "../logistic-titanic-data/titanic-clean.csv"), ',', String, skipstart=1)
const N_OBS = size(_raw, 1)

# Columns: Survived(1), Pclass(2), Sex(3), Age(4), SibSp(5), Parch(6), Fare(7)
const y = [parse(Int, _raw[i, 1]) for i in 1:N_OBS]

const X = let
    pclass   = [parse(Int, _raw[i, 2]) for i in 1:N_OBS]
    sex_male = [_raw[i, 3] == "male" ? 1.0 : 0.0 for i in 1:N_OBS]
    age      = [parse(Float64, _raw[i, 4]) for i in 1:N_OBS]
    sibsp    = [parse(Float64, _raw[i, 5]) for i in 1:N_OBS]
    parch    = [parse(Float64, _raw[i, 6]) for i in 1:N_OBS]
    fare     = [parse(Float64, _raw[i, 7]) for i in 1:N_OBS]
    hcat(
        [pclass[i] == 2 ? 1.0 : 0.0 for i in 1:N_OBS],  # Pclass_2
        [pclass[i] == 3 ? 1.0 : 0.0 for i in 1:N_OBS],  # Pclass_3
        sex_male,
        age,
        sibsp,
        parch,
        fare,
    )
end

const N_COV  = 7      
const N_BETA = N_COV + 1 
const N_DIM  = N_BETA  
const σ_prior = 10

# Normal titanic model for linear path
struct TitanicModel <: NRPT.SamplingProblem end

function NRPT.V0(::TitanicModel, params)
    for j in 1:N_DIM
        lp += logpdf(Cauchy(0.0, σ_prior), params[j])
    end
    return lp
end

NRPT.V1(m::TitanicModel, params) = NRPT.V0(m, params) + logistic_loglik(params)

function logistic_loglik(params)
    β0 = params[1]
    β1 = @view params[2:end]
    lsum = 0.0
    @inbounds for i in 1:N_OBS
        η = β0 + dot(β1, @view(X[i, :]))
        lsum += y[i] * η - log1pexp(η)
    end
    return isfinite(lsum) ? lsum : -Inf
end

function NRPT.sample_iid(::TitanicModel)
    β = rand(Cauchy(0.0, σ_prior), N_BETA)
    return β
end

function NRPT.sample_iid!(::TitanicModel, x)
    for j in 1:N_DIM
        x[j] = rand(Cauchy(0.0, σ_prior))
    end
    return x
end

# Prior reparametrized path
struct TitanicGBM <: NRPT.GBM end

Base.length(::TitanicGBM) = N_DIM

function NRPT.T(::TitanicGBM, z)
    out = similar(z)
    for j in 1:N_BETA
        u        = cdf(Normal(0.0, 1.0), z[j])
        out[j] = quantile(Cauchy(0.0, σ_prior), u)
    end
    return out
end

struct TitanicLikelihood <: NRPT.Likelihood end

NRPT.loglik(::TitanicLikelihood, x) = logistic_loglik(x)

const titanic_gbm = TitanicGBM()
