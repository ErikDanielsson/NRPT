abstract type SamplingProblem end
log_potentials(problem::SamplingProblem, x) = [V0(problem, x), V1(problem, x)]
abstract type Path end
abstract type Explorer end
struct PathProblem{T <: SamplingProblem, P <: Path, E <: Explorer} 
    problem::T
	path::P
	explorer::E
end

function step(problem::PathProblem, x, β)
    return (β == 0.0
        ? sample_iid(problem.problem)
        : step(problem.explorer, problem, x, β)
    )
end

log_potential(problem::PathProblem, x, β) = log_potential(problem.path, log_potentials(problem.problem, x), β)
log_potentials(problem::PathProblem, x) = log_potentials(problem.problem, x)

function run_single_chain(x::T, problem::PathProblem, β, n::Int) where {T}  
    xs = Vector{T}(undef, n)
    for i in 1:n-1
        xs[i] = x
        x = step(problem, x, β)
    end
    xs[n] = x
    return xs
end
