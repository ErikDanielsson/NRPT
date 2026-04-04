# Bidirectional perturbed linear path:
#   W_β(x) = (1-β)·V0 + β·V1 + (1-β)·β·[∑ᵢ cᵢ·softplus(V1-V0-10i) + ∑ᵢ dᵢ·softplus(V0-V1-10i)]
#
# Parameters are split into two blocks of length N:
#   t[1:N]   → c  (forward perturbations)
#   t[N+1:2N] → d  (backward perturbations)
#
# Each block uses the same param_to_c / c_to_param reparametrisation
# from PerturbedLinearPathNoProj.jl (∑ coefficients > 0, unconstrained
# free parameters, full-rank Jacobian, default t=0 gives cᵢ=dᵢ=1/N).

mutable struct PerturbedLinearPathBidir{T<:AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

function PerturbedLinearPathBidir{T}(c0::T, d0::T, backend::AbstractADType) where {T<:AbstractVector{<:Real}}
    @assert length(c0) == length(d0)
    t0 = [c_to_param(c0); c_to_param(d0)]
    return PerturbedLinearPathBidir(t0, nothing, backend)
end

# Convenience constructor: default initialisation t = 0 (cᵢ = dᵢ = 1/N)
PerturbedLinearPathBidir(N::Int, backend::AbstractADType) =
    PerturbedLinearPathBidir{Vector{Float64}}(zeros(N), zeros(N), backend)

(path::PerturbedLinearPathBidir)(t, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        N = length(t) ÷ 2
        c = param_to_c(t[1:N])
        d = param_to_c(t[N+1:2N])
        forward  = sum(c[i] * softplus(V1 - V0 - 10(i - 1)) for i in eachindex(c))
        backward = sum(d[i] * softplus(V0 - V1 - 10(i - 1)) for i in eachindex(d))
        return V0 * (1 - β) + β * V1 + (1 - β) * β * (forward + backward)
    end
end

function log_potential(path::PerturbedLinearPathBidir, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

extract_param(path::PerturbedLinearPathBidir) = path.t

function extract_reparam(path::PerturbedLinearPathBidir)
    N = length(path.t) ÷ 2
    return [param_to_c(path.t[1:N]); param_to_c(path.t[N+1:2N])]
end

function set_param!(path::PerturbedLinearPathBidir, t::T) where {T <: AbstractVector}
    path.t = t
end
