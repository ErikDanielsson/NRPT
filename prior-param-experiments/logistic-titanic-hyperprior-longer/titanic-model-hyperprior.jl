using Distributions, NRPT, Random, LinearAlgebra, DelimitedFiles
using LogExpFunctions

# --- Data loading and preprocessing ---

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

const EXP_PRIOR_MEAN = 10.

const N_COV  = 7           
const N_BETA = N_COV + 1   
const N_DIM  = N_BETA + 1  

struct TitanicModel <: NRPT.SamplingProblem end

function NRPT.V0(::TitanicModel, params)
    σ = params[1]
    lp = logpdf(Exponential(EXP_PRIOR_MEAN), σ)  # log Exp(1)(σ)
    for j in 2:N_DIM
        lp += logpdf(Cauchy(0.0, σ), params[j])
    end
    return lp
end

NRPT.V1(m::TitanicModel, params) = NRPT.V0(m, params) + logistic_loglik(params)

function logistic_loglik(params)
    β0 = params[2]
    β1 = @view params[3:end]
    lsum = 0.0
    @inbounds for i in 1:N_OBS
        η = β0 + dot(β1, @view(X[i, :]))
        lsum += y[i] * η - log1pexp(η)
    end
    return isfinite(lsum) ? lsum : -Inf
end

function NRPT.sample_iid(::TitanicModel)
    β = rand(Cauchy(0.0, σ), N_BETA)
    return vcat([σ], β)
end

function NRPT.sample_iid!(::TitanicModel, x)
    σ = rand(Exponential(EXP_PRIOR_MEAN))
    x[1] = σ
    for j in 2:N_DIM
        x[j] = rand(Cauchy(0.0, σ))
    end
    return x
end

struct TitanicGBM <: NRPT.GBM end

Base.length(::TitanicGBM) = N_DIM

function NRPT.T(::TitanicGBM, z)
    ε = eps(Float64)
    u1 = clamp(cdf(Normal(0.0, 1.0), z[1]), ε, 1.0 - ε)
    σ  = quantile(Exponential(EXP_PRIOR_MEAN), u1) 
    out = similar(z)
    out[1] = σ
    for j in 1:N_BETA
        u        = clamp(cdf(Normal(0.0, 1.0), z[j + 1]), ε, 1.0 - ε)
        out[j+1] = quantile(Cauchy(0.0, σ), u)
    end
    return out
end

struct TitanicLikelihood <: NRPT.Likelihood end

NRPT.loglik(::TitanicLikelihood, x) = logistic_loglik(x)

const titanic_gbm = TitanicGBM()
