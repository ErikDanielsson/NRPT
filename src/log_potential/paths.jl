abstract type Path{P<:SamplingProblem} end

log_potential(::Path, x, β) = throw(MethodError(log_potential, x, β))
get_problem(::Path) = throw(MethodError(log_potential, x, β))

# Get exponentens from a linear like path. This is only well defined
# on certain paths, but is useful for sampling iid in exponential families
get_exponents(::Path, β) = throw(MethodError(log_potential, β))

abstract type StaticPath{P<:SamplingProblem} <: Path{P} end

abstract type ParametrizedPath{T, P<:SamplingProblem} <: Path{P} end

gradient(::ParametrizedPath, x, β) = throw(MethodError(gradient, x, β))
extract_param(::ParametrizedPath, x, β) = throw(MethodError(extract_param, x, β))
extract_reparam(::ParametrizedPath, x, β) = throw(MethodError(extract_reparam, x, β))
set_param!(::ParametrizedPath, param) = throw(MethodError(set_param!, param))

# Functions for preparing and computing derivatives of paths

function prepare_path_gradient(log_potential::Function, params::Float64, xlike::T, backend::AbstractADType) where {T}
    return DifferentiationInterface.prepare_derivative(
        log_potential,
        backend,
        params,
        DifferentiationInterface.Constant(xlike),
        DifferentiationInterface.Constant(1.0)
    )
end

function prepare_path_gradient(log_potential::Function, params, xlike::T, backend::AbstractADType) where {T}
    return DifferentiationInterface.prepare_gradient(
        log_potential,
        backend,
        similar(params),
        DifferentiationInterface.Constant(xlike),
        DifferentiationInterface.Constant(1.0)
    )
end

function path_gradient(log_potential, prep, params::Float64, x, β, backend)
    return DifferentiationInterface.derivative(
        log_potential,
        prep,
        backend,
        params,
        DifferentiationInterface.Constant(x),
        DifferentiationInterface.Constant(β)
    )
end

function path_gradient(log_potential, prep, params, x, β, backend)
    return DifferentiationInterface.gradient(
        log_potential,
        prep,
        backend,
        params,
        DifferentiationInterface.Constant(x),
        DifferentiationInterface.Constant(β)
    )
end