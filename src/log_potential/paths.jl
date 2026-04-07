
log_potential(::Path, base_potentials, β) = throw(MethodError(log_potential, x, β))

# Get exponents from a linear like path. This is only well defined
# on certain paths, but is useful for sampling iid in exponential families
get_exponents(::Path, β) = throw(MethodError(log_potential, β))

abstract type StaticPath <: Path end

abstract type ParametrizedPath{T} <: Path end

extract_param(::ParametrizedPath) = throw(MethodError(extract_param, x, β))
extract_reparam(::ParametrizedPath) = throw(MethodError(extract_reparam, x, β))
set_param!(::ParametrizedPath, param) = throw(MethodError(set_param!, param))

# Generic gradient: path itself is the DI callable.
# Each ParametrizedPath must implement (path::MyPath)(params, lps, β) as its call operator.
# Prep is computed lazily on the first call using the actual argument types, then cached.
function gradient(path::P, base_potentials::AbstractVector{Float64}, β) where {P <: ParametrizedPath}
    if isnothing(path.prep)
        path.prep = _prepare_path_gradient(path, extract_param(path), base_potentials, β, path.backend)
    end
    return _path_gradient(path, path.prep, extract_param(path), base_potentials, β, path.backend)
end

function _prepare_path_gradient(path, params::Float64, base_potentials, β, backend::AbstractADType)
    return DifferentiationInterface.prepare_derivative(
        path, backend, params,
        DifferentiationInterface.Constant(base_potentials),
        DifferentiationInterface.Constant(β)
    )
end

function _prepare_path_gradient(path, params, base_potentials, β, backend::AbstractADType)
    return DifferentiationInterface.prepare_gradient(
        path, backend, copy(params),
        DifferentiationInterface.Constant(base_potentials),
        DifferentiationInterface.Constant(β)
    )
end

function _path_gradient(path, prep, params::Float64, base_potentials::AbstractVector{Float64}, β, backend)
    return DifferentiationInterface.derivative(
        path, prep, backend, params,
        DifferentiationInterface.Constant(base_potentials),
        DifferentiationInterface.Constant(β)
    )
end

function _path_gradient(path, prep, params, base_potentials::AbstractVector{Float64}, β, backend)
    return DifferentiationInterface.gradient(
        path, prep, backend, params,
        DifferentiationInterface.Constant(base_potentials),
        DifferentiationInterface.Constant(β)
    )
end
