mutable struct ConstantPerturbedLinearPath{T <: AbstractVector{<:Real}} <: ParametrizedPath{T}
    t::T
    basis
    prep
    backend::AbstractADType
end

function ConstantPerturbedLinearPath(n_knots::Int, backend::AbstractADType)
    basis = collect(BSplineKit.BSplineBasis(BSplineKit.BSplineOrder(4), range(-100, 100, n_knots)))[2:(end - 1)]
    t0 = zeros(n_knots)
    return ConstantPerturbedLinearPath(t0, basis, nothing, backend)
end

(path::ConstantPerturbedLinearPath)(t, log_potentials::AbstractVector{Float64}, β) = begin
    V0, V1 = log_potentials
    if β == 0.0
        return V0
    elseif β == 1.0
        return V1
    else
        perturbation = sum(ti .* bi(V1 - V0) for (ti, bi) in zip(t, path.basis))
        return V0 * (1 - β) + β * V1 + β * (1 - β) * perturbation
    end
end

function log_potential(path::ConstantPerturbedLinearPath, log_potentials::AbstractVector{Float64}, β)
    return path(path.t, log_potentials, β)
end

extract_param(path::ConstantPerturbedLinearPath) = path.t
extract_reparam(path::ConstantPerturbedLinearPath) = path.t

function set_param!(path::ConstantPerturbedLinearPath, t::T) where {T <: AbstractVector}
    return path.t = t
end
