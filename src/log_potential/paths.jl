
log_potential(::Path, log_potentials::AbstractVector{Float64}, β) = throw(MethodError(log_potential, x, β))

# Get exponentens from a linear like path. This is only well defined
# on certain paths, but is useful for sampling iid in exponential families
get_exponents(::Path, β) = throw(MethodError(log_potential, β))

abstract type StaticPath <: Path end

abstract type ParametrizedPath{T} <: Path end

gradient(::ParametrizedPath, log_potentials::AbstractVector{Float64}, β) = throw(MethodError(gradient, x, β))
extract_param(::ParametrizedPath) = throw(MethodError(extract_param, x, β))
extract_reparam(::ParametrizedPath) = throw(MethodError(extract_reparam, x, β))
set_param!(::ParametrizedPath, param) = throw(MethodError(set_param!, param))

# Functions for preparing and computing derivatives of paths

function prepare_path_gradient(log_potential::Function, params::Float64, backend::AbstractADType)
    return DifferentiationInterface.prepare_derivative(
        log_potential,
        backend,
        params,
        DifferentiationInterface.Constant(zeros(2)), # Log potentials, we only handle two right now
        DifferentiationInterface.Constant(1.0) # β
    )
end

function prepare_path_gradient(log_potential::Function, params, backend::AbstractADType)
    return DifferentiationInterface.prepare_gradient(
        log_potential,
        backend,
        similar(params),
        DifferentiationInterface.Constant(zeros(2)), # Log potentials, we only handle two right now
        DifferentiationInterface.Constant(1.0), # β
    )
end

function path_gradient(log_potential, prep, params::Float64, log_potentials::AbstractVector{Float64}, β, backend)
    return DifferentiationInterface.derivative(
        log_potential,
        # prep,
        backend,
        params,
        DifferentiationInterface.Constant(log_potentials),
        DifferentiationInterface.Constant(β)
    )
end

function path_gradient(log_potential, prep, params, log_potentials::AbstractVector{Float64}, β, backend)
    return DifferentiationInterface.gradient(
        log_potential,
        # prep,
        backend,
        params,
        DifferentiationInterface.Constant(log_potentials),
        DifferentiationInterface.Constant(β)
    )
end