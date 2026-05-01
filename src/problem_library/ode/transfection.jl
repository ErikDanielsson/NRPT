# Transfection ODE model from Raimundez et al.
# Parameters x = [log10(km0), log10(δ), log10(β), log10(t0), log10(σ)] ∈ ℝ⁵.
# Prior: LogUniform on each axis (uniform on log10 scale within box).
# Likelihood: observations O_t ~ N(ode_mean(t, ...), σ²).

const _TRANSFECTION_PRIOR_LB = [-5.0, -5.0, -5.0, -2.0, -2.0]
const _TRANSFECTION_PRIOR_UB = [ 5.0,  5.0,  5.0,  1.0,  2.0]

function _transfection_ode_mean(t, km0, δ, β, t0)
    Δ = δ - β
    return km0 / Δ * (1 - exp(-Δ * (t - t0))) * exp(-β * (t - t0))
end

# data_matrix: N×2 matrix with columns [time, observation].
function transfection_ode_slice_sampler(data_matrix::AbstractMatrix; steps=5)
    dimension = 5
    data = [(data_matrix[i, 1], data_matrix[i, 2]) for i in axes(data_matrix, 1)]

    function log_prior(x)
        for i in eachindex(x)
            (x[i] < _TRANSFECTION_PRIOR_LB[i] || x[i] > _TRANSFECTION_PRIOR_UB[i]) && return -Inf
        end
        return zero(eltype(x))
    end

    function sample_prior()
        return [rand() * (_TRANSFECTION_PRIOR_UB[i] - _TRANSFECTION_PRIOR_LB[i]) + _TRANSFECTION_PRIOR_LB[i]
                for i in 1:dimension]
    end

    function log_likelihood(x, d)
        t, O = d
        km0 = 10^x[1]; δ = 10^x[2]; β = 10^x[3]; t0 = 10^x[4]; σ = 10^x[5]
        μ = _transfection_ode_mean(t, km0, δ, β, t0)
        if isnan(μ) || isinf(μ); μ = 10_000.0; end
        return logpdf(Normal(μ, σ), O)
    end

    return (
        PosteriorProblem(log_prior, sample_prior, log_likelihood, data),
        IterExplorer(SliceSampler(), steps)
    )
end

# Load transfection data from a CSV (columns: times, sample, observations; has header).
# Returns an N×2 matrix [times observations].
function load_transfection_data(path)
    raw = readdlm(path, ',', Float64; skipstart=1)
    return hcat(raw[:, 1], raw[:, 3])
end

struct TransfectionLikelihood <: Likelihood
    data::Matrix{Float64}
end

function loglik(l::TransfectionLikelihood, x)
    km0 = 10^x[1]; δ = 10^x[2]; β_rate = 10^x[3]; t0 = 10^x[4]; σ = 10^x[5]
    ll = 0.0
    for i in axes(l.data, 1)
        t, obs = l.data[i, 1], l.data[i, 2]
        μ = _transfection_ode_mean(t, km0, δ, β_rate, t0)
        (isnan(μ) || isinf(μ)) && (μ = 10000.0)
        ll += logpdf(Normal(μ, σ), obs)
    end
    return ll
end

function transfection_ode_gbm(data::AbstractMatrix; steps=5)
    gbm = BoundedUniformGBM(_TRANSFECTION_PRIOR_LB, _TRANSFECTION_PRIOR_UB)
    lik = TransfectionLikelihood(Matrix{Float64}(data))
    sp  = GBMProblem(gbm, lik)
    explorer = IterExplorer(SliceSampler(), steps)
    return sp, explorer
end
