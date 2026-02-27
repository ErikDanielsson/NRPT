struct SamplingProblem{T, S, W, L, D}
    V0::T
    sample_iid::S
    V::W
    data::Vector{D}
    V1::L
end

function SamplingProblem(V0, sample_iid, V, data)
    function V1(x)
        sum(V(x, d) for d in data)
    end
    return SamplingProblem(
        V0, sample_iid, V, data, V1
    )
end