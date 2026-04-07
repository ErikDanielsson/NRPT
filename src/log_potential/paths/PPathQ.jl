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

mutable struct PPathQ{T<:AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    prep
    backend::AbstractADType
end

PPathQ(N::Int, backend::AbstractADType) = PPathQ([1.; zeros(N - 1)], nothing, backend)

(path::PPathQ)(t, log_potentials::LP, β) where {LP <: AbstractVector{Float64}} = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        c = @view(softmax(t)[2:end])
        N = length(c)
        # println([N / i * softplus(i / N * (V0 - V1)) for i in 1:N])
        forward  = sum(c[i] * N / i * softplus(i / N * (V0 - V1)) for i in 1:N)
        return V0 * (1 - β) + β * V1 + (1 - β) * β * forward
    end
end

function log_potential(path::PPathQ, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

extract_param(path::PPathQ) = path.t

function extract_reparam(path::PPathQ)
    return softmax(path.t)[2:end]
end

function set_param!(path::PPathQ, t::T) where {T <: AbstractVector}
    path.t = t
end
