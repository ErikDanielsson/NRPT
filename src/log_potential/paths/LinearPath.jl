struct LinearPath <: StaticPath
	log_potential
end

function LinearPath()
    function __log_potential(log_potentials::Vector{Float64}, β)
        V0, V1 = log_potentials
        return (1 - β) * V0 + β * V1
    end
    return LinearPath(__log_potential) 
end

function log_potential(path::LinearPath, log_potentials::Vector{Float64}, β::T) where {T <: Real}
    return path.log_potential(log_potentials, β)
end

get_exponents(::LinearPath, β) = (1 - β, β)