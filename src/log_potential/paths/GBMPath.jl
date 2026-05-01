abstract type GBMPath{C} <: ParametrizedPath{C} end

mutable struct ScalingBaseMeasureChange{C <: Real} <: BaseMeasureChange
    c::C
end

V0β(c::Real, gbm::GBM, z::AbstractVector) = V0β(c, V0(gbm, z))
V0β(σ::Real, V0) = V0 / σ
V0β(sbcm::ScalingBaseMeasureChange, gbm::GBM, z::AbstractVector) = V0β(sbcm.c, gbm, z)
V0β(sbcm::ScalingBaseMeasureChange, V0) = V0β(sbcm.c, V0)

mutable struct ScalingGBMPath{C <: AbstractVector{<:Real}, P <: Path} <: GBMPath{C}
    c::C
    path::P
    basis::BernsteinBasis
    prep
    backend::AbstractADType
end

function c_to_σ(path::ScalingGBMPath, c, β)
    return exp(path.basis(c, β))
end

function c_to_σ(path::ScalingGBMPath, β)
    c_to_σ(path, path.c, β)
end

function ScalingGBMPath(order::Int, path::P, backend::AbstractADType) where {P <: Path}
    basis, c0 = generate_basis_and_vector(order)
    return ScalingGBMPath{Vector{Float64}, P}(c0, path, basis, nothing, backend)
end

# # Convenience constructors without explicit type parameters
# ScalingGBMPath(c0::C, path::P, backend::AbstractADType) where {C <: Real, P <: Path} =
#     ScalingGBMPath{C, P}(c0, path, nothing, backend)

# ScalingGBMPath(path::P, backend::AbstractADType) where {P <: Path} =
#     ScalingGBMPath(1.0, path, backend)



# ── Callable: StaticPath case ────────────────────────────────────────────────
# extract_param returns c::Float64; AD differentiates w.r.t. c only.
function (path::ScalingGBMPath{C, <:StaticPath})(
        c::AbstractVector, lps::AbstractVector{<:Real}, β) where {C}
    β == 0.0 && return lps[1]
    β == 1.0 && return lps[2]
    V0_new = V0β(c_to_σ(path, c, β), lps[1])
    log_potential(path.path, [V0_new, lps[2]], β)
end

# ── Callable: ParametrizedPath case ─────────────────────────────────────────
# extract_param returns [c; inner_params]::Vector; AD differentiates w.r.t. all.
# The inner path may expect a scalar (e.g. QPath) or a vector (e.g. PerturbedLinearPath),
# so we dispatch on the inner path's parameter type to extract the right shape.
_inner_param(t::AbstractVector, ::ParametrizedPath{<:Real})           = t[1]
_inner_param(t::AbstractVector, ::ParametrizedPath{<:AbstractVector}) = t

function (path::ScalingGBMPath{C, <:ParametrizedPath})(
        t::AbstractVector, lps::AbstractVector{<:Real}, β) where {C}
    β == 0.0 && return lps[1]
    β == 1.0 && return lps[2]
    n_params = length(path.c)
    outer_t = @view(t[1:n_params])
    inner_t = _inner_param(@view(t[n_params+1:end]), path.path)
    V0_new  = V0β(c_to_σ(path, outer_t, β), lps[1])
    path.path(inner_t, [V0_new, lps[2]], β)
end

function log_potential(path::ScalingGBMPath, lps::AbstractVector{<:Real}, β::Real)
    path(extract_param(path), lps, β)
end

# ── Parameter accessors ──────────────────────────────────────────────────────
extract_param(path::ScalingGBMPath{C, <:StaticPath}) where {C} = path.c
extract_param(path::ScalingGBMPath{C, <:ParametrizedPath}) where {C} =
    [path.c; extract_param(path.path)]

extract_reparam(path::ScalingGBMPath) = extract_param(path)

function set_param!(path::ScalingGBMPath{C, <:StaticPath}, c::C) where {C}
    path.c = c
end

function set_param!(path::ScalingGBMPath{C, <:ParametrizedPath}, t::AbstractVector) where {C}
    n_params = length(path.c)
    outer_t = @view(t[1:n_params])
    inner_t = _inner_param(@view(t[n_params+1:end]), path.path)
    path.c = outer_t
    set_param!(path.path, _inner_param(inner_t, path.path))
end
