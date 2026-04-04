struct LinearPath <: StaticPath end

function log_potential(::LinearPath, log_potentials::AbstractVector{Float64}, β::T) where {T <: Real}
    if β == 0.0
        return log_potentials[1]
    elseif β == 1.0
        return log_potentials[2]
    else
        V0, V1 = log_potentials
        return (1 - β) * V0 + β * V1
    end
end

get_exponents(::LinearPath, β) = [1 - β, β]

extract_reparam(::LinearPath) = nothing
extract_param(::LinearPath) = nothing
