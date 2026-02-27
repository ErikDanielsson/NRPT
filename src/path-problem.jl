struct PathProblem{P <: Path, E <: Explorer} 
	path::P
	explorer::E
end

function step(problem::PathProblem, x, β)
    return (β == 0.0
        ? sample_iid(problem.path)
        : step(problem.explorer, problem.path, x, β)
    )
end

function run_single_chain(x::T, problem::PathProblem, β, n::Int) where {T}  
    xs = Vector{T}(undef, n)
    for i in 1:n-1
        xs[i] = x
        x = step(problem, x, β)
    end
    xs[n] = x
    return xs
end
