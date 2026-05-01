abstract type SamplingProblem end
base_potentials(problem::SamplingProblem, x::T) where {T} = [V0(problem, x), V1(problem, x)]
function base_potentials!(problem::SamplingProblem, x::T, lp_buff::LP) where {T, LP <: AbstractVector{Float64}}
    lp_buff[1] = V0(problem, x)
    lp_buff[2] = V1(problem, x)
    return lp_buff
end
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

function step!(problem::PathProblem, x, β)
    return (β == 0.0
        ? sample_iid!(problem.problem, x)
        : step!(problem.explorer, problem, x, β)
    )
end

# log_potential(problem::PathProblem, x::T, β::Float64) where {T} = log_potential(problem.path, log_potentials(problem.problem, x), β)
log_potential!(problem::PathProblem, x::T, β::Float64, lp_buff::LP) where {T, LP <: AbstractVector{Float64}} =
    log_potential(problem.path, base_potentials!(problem.problem, x, lp_buff), β)

# log_potentials(problem::PathProblem, x) = log_potentials(problem.problem, x)
base_potentials!(problem::PathProblem, x::T, lp_buff::LP) where {T, LP <: AbstractVector{Float64}} =
    base_potentials!(problem.problem, x, lp_buff)

function step(problem::PathProblem, x, β, lp_buff::AbstractVector{Float64})
    return (β == 0.0
        ? sample_iid(problem.problem)
        : step(problem.explorer, problem, x, β, lp_buff)
    )
end

function step!(problem::PathProblem, x, β, lp_buff::AbstractVector{Float64})
    return (β == 0.0
        ? sample_iid!(problem.problem, x)
        : step!(problem.explorer, problem, x, β, lp_buff)
    )
end

function run_single_chain(x::T, problem::PathProblem, β, n::Int) where {T}  
    xs = Vector{T}(undef, n)
    lp_buff = zeros(2)
    for i in 1:n-1
        xs[i] = copy(x)
        x = step(problem, x, β, lp_buff)
    end
    xs[n] = x
    return xs
end
