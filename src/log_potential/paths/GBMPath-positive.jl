abstract type GBMPath{C} <: ParametrizedPath{C} end

mutable struct ScalingBaseMeasureChange{C <: Real} <: BaseMeasureChange
    c::C
end

V0β(c::Real, gbm::GBM, z::AbstractVector) = V0β(c, V0(gbm, z))
V0β(τ::Real, V0) = V0 * τ
V0β(sbcm::ScalingBaseMeasureChange, gbm::GBM, z::AbstractVector) = V0β(sbcm.c, gbm, z)
V0β(sbcm::ScalingBaseMeasureChange, V0) = V0β(sbcm.c, V0)

mutable struct ScalingGBMPath{C <: AbstractVector{<:Real}, P <: Path} <: GBMPath{C}
    c::C
    path::P
    basis::Bernstein
    prep
    backend::AbstractADType
end

function c_to_τ(path::ScalingGBMPath, c, β)
    return path.basis(expc, β)
end

function c_to_τ(path::ScalingGBMPath, β)
    return c_to_τ(path, path.c, β)
end

function get_τ0(path::ScalingGBMPath)
    return c_to_τ(path, 0.0)
end

function ScalingGBMPath(order::Int, endpoint::Bool, path::P, backend::AbstractADType) where {P <: Path}
    basis, c0 = generate_basis_and_vector(order, endpoint)
    println(c0)
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
        c::AbstractVector, lps::AbstractVector{<:Real}, β
    ) where {C}
    if β == 1.0
        return lps[2]
    end
    V0_new = V0β(c_to_τ(path, c, β), lps[1])
    if β == 0.0
        return V0_new
    end
    return log_potential(path.path, [V0_new, lps[2]], β)
end

# ── Callable: ParametrizedPath case ─────────────────────────────────────────
# extract_param returns [c; inner_params]::Vector; AD differentiates w.r.t. all.
# The inner path may expect a scalar (e.g. QPath) or a vector (e.g. PerturbedLinearPath),
# so we dispatch on the inner path's parameter type to extract the right shape.
_inner_param(t::AbstractVector, ::ParametrizedPath{<:Real}) = t[1]
_inner_param(t::AbstractVector, ::ParametrizedPath{<:AbstractVector}) = t

function (path::ScalingGBMPath{C, <:ParametrizedPath})(
        t::AbstractVector, lps::AbstractVector{<:Real}, β
    ) where {C}
    if β == 1.0
        return lps[2]
    end
    n_params = length(path.c)
    outer_t = @view(t[1:n_params])
    V0_new = V0β(c_to_τ(path, outer_t, β), lps[1])
    if β == 0.0
        return V0_new
    end
    inner_t = _inner_param(@view(t[(n_params + 1):end]), path.path)
    return path.path(inner_t, [V0_new, lps[2]], β)
end

function log_potential(path::ScalingGBMPath, lps::AbstractVector{<:Real}, β::Real)
    return path(extract_param(path), lps, β)
end

# ── Parameter accessors ──────────────────────────────────────────────────────
extract_param(path::ScalingGBMPath{C, <:StaticPath}) where {C} = path.c
extract_param(path::ScalingGBMPath{C, <:ParametrizedPath}) where {C} =
    [path.c; extract_param(path.path)]

extract_reparam(path::ScalingGBMPath) = extract_param(path)

function set_param!(path::ScalingGBMPath{C, <:StaticPath}, c::C) where {C}
    # @info "τ0 = $(get_τ0(path))"
    # @info "τ0.5 = $(c_to_τ(path, 0.5))"
    return path.c = c
end

function set_param!(path::ScalingGBMPath{C, <:ParametrizedPath}, t::AbstractVector) where {C}
    n_params = length(path.c)
    outer_t = @view(t[1:n_params])
    inner_t = _inner_param(@view(t[(n_params + 1):end]), path.path)
    path.c[:] = outer_t
    return set_param!(path.path, _inner_param(inner_t, path.path))
end

# If we are using a ScalingGBM path then we might be using a different reference measure
function step!(problem::PathProblem{P, <:ScalingGBMPath, E}, x, β, lp_buff::AbstractVector{Float64}) where {P <: SamplingProblem, E <: Explorer}
    return if β == 0.0
        sample_iid!(problem.problem, x)
        x[:] ./= sqrt(get_τ0(problem.path))
    else
        step!(problem.explorer, problem, x, β, lp_buff)
    end
end
