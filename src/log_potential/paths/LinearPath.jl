struct LinearPath <: StaticPath end

function log_potential(::LinearPath, log_potentials::AbstractVector{<:Real}, β::T) where {T <: Real}
    if β == 0.0
        @inbounds return log_potentials[1]
    elseif β == 1.0
        @inbounds return log_potentials[2]
    else
        @inbounds V0 = log_potentials[1]
        @inbounds V1 = log_potentials[2]
        return (1 - β) * V0 + β * V1
    end
end

get_exponents(::LinearPath, β) = [1 - β, β]

extract_reparam(::LinearPath) = nothing
extract_param(::LinearPath) = nothing
