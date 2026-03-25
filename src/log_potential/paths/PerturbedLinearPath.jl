# Perturbed linear path:
#   W_β(x) = (1-β)·V0 + β·V1 + (1-β)·β·∑ᵢ₌₁ᴺ cᵢ·log(10ⁱ + exp(V1-V0))
#
# Natural parametrisation for the constraint ∑ cᵢ ≥ 0:
#
#   Decompose c = (s/n)·𝟏 + d  where  s = ∑ cᵢ = exp(t₀) > 0  and  ∑ dᵢ = 0.
#   Free parameters: t = [t₀, u₁, …, u_{n-1}] ∈ ℝⁿ
#
#       cᵢ = exp(t₀)/n + uᵢ   for i = 1, …, n-1
#       cₙ = exp(t₀)/n - ∑ᵢ₌₁ⁿ⁻¹ uᵢ
#
#   The sum-direction and zero-mean-deviation directions are orthogonal, so
#   the Jacobian ∂c/∂t is full-rank everywhere with no conditioning issues.
#   Default initialisation t = 0 gives cᵢ = 1/n for all i and ∑ cᵢ = 1.

function param_to_c(t::AbstractVector)
    n = length(t)
    mean_c = exp(t[1]) / n   # s/n where s = exp(t₀) = ∑ cᵢ > 0
    u = t[2:end]              # length n-1, free deviation parameters
    c = Vector{eltype(t)}(undef, n)
    c[1:n-1] .= mean_c .+ u
    c[n] = mean_c - sum(u)
    return c
end

# Inverse: requires ∑ cᵢ > 0 (interior of feasible set).
function c_to_param(c::AbstractVector)
    n = length(c)
    s = sum(c)
    t0 = log(s)               # s = exp(t₀), so t₀ = log(∑ cᵢ)
    mean_c = s / n
    u = c[1:n-1] .- mean_c
    return [t0; u]
end

mutable struct PerturbedLinearPath{T<:AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    log_potential::Function
    prep
    backend::AbstractADType
end

function PerturbedLinearPath(c0::AbstractVector, backend::AbstractADType)
    # Use c_to_param when c0 is in the feasible interior; otherwise use the
    # default t = 0 initialisation (cᵢ = 1/N for all i, ∑ cᵢ = 1).
    function __log_potential(c, log_potentials::AbstractVector{Float64}, β)
        V0, V1 = log_potentials
        if β == 0.0
            return V0
        elseif β == 1.0
            return V1
        else
            perturbation = c[1] + sum(c[i + 1] * softplus(V1 - V0 - 10i) for i in eachindex(c[2:end]))
            return V0 * (1 - β) + β * V1 + (1 - β) * β * perturbation
        end
    end
    prep = prepare_path_gradient(__log_potential, c0, backend)
    return PerturbedLinearPath(c0, __log_potential, prep, backend)
end

# Convenience constructor: default initialisation t = 0 (cᵢ = 1/N)
PerturbedLinearPath(N::Int, backend::AbstractADType) =
    PerturbedLinearPath(zeros(N), backend)

function log_potential(path::PerturbedLinearPath, log_potentials::AbstractVector{Float64}, β)
    return path.log_potential(path.t, log_potentials, β)
end

function gradient(path::PerturbedLinearPath, log_potentials::AbstractVector{Float64}, β)
    return path_gradient(path.log_potential, path.prep, path.t, log_potentials, β, path.backend)
end

extract_param(path::PerturbedLinearPath) = path.t
extract_reparam(path::PerturbedLinearPath) = path.t 

function set_param!(path::PerturbedLinearPath, t::T) where {T <: AbstractVector}
    path.t = t
end
