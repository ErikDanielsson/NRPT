mutable struct PotentialRecorder
    base_potentials::Array{Float64, 3}
end

function record_base_potentials!(base_potentials, j::Int)
    @inbounds chain.base_potentials[:, iteration, j] = base_potentials
end