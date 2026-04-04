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

function param_to_c(t::AbstractVector; b=1.0)
    n = length(t)
    s = b - softplus(t[1])    # s/n where s = exp(t₀) = ∑ cᵢ > 0
    u = t[2:end]              # length n-1, free deviation parameters
    c = Vector{eltype(t)}(undef, n)
    c[2:n] = u
    c[1] = s - sum(u)
    return c
end

# Inverse: requires ∑ cᵢ > 0 (interior of feasible set).
function c_to_param(c::AbstractVector; b=1.0)
    n = length(c)
    s = sum(c)
    t0 = invsoftplus(b - s)
    u = c[2:n]
    return [t0; u]
end

mutable struct PerturbedLinearPath{T<:AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

function PerturbedLinearPath(c0::AbstractVector, backend::AbstractADType)
    t0 = c_to_param(c0)
    return PerturbedLinearPath(t0, nothing, backend)
end

# Convenience constructor: default initialisation t = 0 (cᵢ = 1/N)
PerturbedLinearPath(N::Int, backend::AbstractADType) =
    PerturbedLinearPath(zeros(N), backend)

(path::PerturbedLinearPath)(t, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        c = param_to_c(t)
        perturbation = sum(c[i] * softplus(V1 - V0 - 10i) for i in eachindex(c[1:end]))
        return V0 * (1 - β) + β * V1 + (1 - β) * β * perturbation
    end
end

function log_potential(path::PerturbedLinearPath, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

extract_param(path::PerturbedLinearPath) = path.t
extract_reparam(path::PerturbedLinearPath) = param_to_c(path.t)

function set_param!(path::PerturbedLinearPath, t::T) where {T <: AbstractVector}
    path.t = t
end
