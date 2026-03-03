struct LinearPath{P<:SamplingProblem} <: StaticPath{P}
	log_potential
    problem::P
end

get_problem(path::LinearPath) = path.problem

function LinearPath(problem::SamplingProblem)
    function __log_potential(x, β)
        return -((1 - β) * V0(problem, x) + β * V1(problem, x))
    end
    return LinearPath(__log_potential, problem) 
end

sample_iid(path::LinearPath) = sample_iid(path.problem)

function log_potential(path::LinearPath, x, β::T) where {T <: Real}
    return path.log_potential(x, β)
end

get_exponents(::LinearPath, β) = (1 - β, β)