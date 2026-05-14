struct LPRecorder{T <: AbstractArray, Inds <: AbstractVector{Int}}
    lps::T
    inds::Inds
end

function LPRecorder(::Val{true}, iterations, n_chains)
    arr = Array{Float64}(undef, n_chains - 1, sum(iterations), 2)
    return LPRecorder(arr, [0; cumsum(iterations)])
end

function record_lps!(recorder::LPRecorder{<:AbstractArray, <:AbstractVector}, round, lps)
    s, e = recorder.inds[round], recorder.inds[round + 1]
    return recorder.lps[:, (s + 1):e, :] = lps
end


function get_round_lps(recorder::LPRecorder{<:AbstractArray, <:AbstractVector}, round)
    s, e = recorder.inds[round], recorder.inds[round + 1]
    return recorder.lps[:, (s + 1):(s + e), :]
end

LPRecorder(::Val{false}, _, _) = nothing

record_lps!(::Nothing, _, _) = nothing

get_round_lps(::Nothing, round) = throw(MethodError(get_round_lps, round))
