struct NonMonotonePath <: StaticPath end

function _nonmonotone_f(β; p = 1)
    return (exp(p * β) - 1) / (exp(p) - 1)
end

function log_potential(::NonMonotonePath, log_potentials::AbstractVector{Float64}, β::T) where {T <: Real}
    if β == 0.0
        return log_potentials[1]
    elseif β == 1.0
        return log_potentials[2]
    else
        V0, V1 = log_potentials
        return V0 + logweightaddexp(1 - β, 1.0, _nonmonotone_f(β), 1 * (V1 - V0)) / 1
    end
end

get_exponents(::NonMonotonePath, β) = [1 - β, β]

extract_reparam(::NonMonotonePath) = nothing
extract_param(::NonMonotonePath) = nothing
