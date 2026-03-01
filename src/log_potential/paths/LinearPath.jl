struct LinearPath <: StaticPath
	log_potential
    sample_iid
end

function LinearPath(problem::SamplingProblem)
    function __log_potential(x, β)
        return -((1 - β) * problem.V0(x) + β * problem.V1(x))
    end
    return LinearPath(__log_potential, problem.sample_iid) 
end

sample_iid(path::LinearPath) = path.sample_iid()

function log_potential(path::LinearPath, x, β::T) where {T <: Real}
    return path.log_potential(x, β)
end

get_exponents(::LinearPath, β) = (1 - β, β)