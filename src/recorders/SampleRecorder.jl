abstract type MaybeSampleRecorder{T} end

mutable struct SampleRecorder{T} <: MaybeSampleRecorder{T}
    xs::Matrix{T}
    rounds::Vector{Int}
    i::Int
end

function make_sample_recorder(record::Bool, n_chains::Int, rounds::Vector{Int}, x0::Vector{T}) where {T}
    if record
        return SampleRecorder{T}(n_chains, rounds, x0)
    else
        return NoSampleRecorder{T}(x0)
    end
end

function SampleRecorder{T}(n_chains::Int, rounds::Vector{Int}, x0::Vector{T}) where {T}
    xs = Matrix{T}(undef, n_chains, sum(rounds))
    xs[:, 1] = copy.(x0)
    return SampleRecorder{T}(xs, rounds, 1)
end

function record!(recorder::SampleRecorder{T}, ptchains::PTChains{N, T, Tr}) where {N, T, Tr}
    recorder.i += 1
    return set_state_per_temperature!(ptchains, @view(recorder.xs[:, recorder.i]))
end

function get_round_samples(recorder::SampleRecorder, round::Int)
    if round == 0
        return @view(recorder.xs[:, 1:1])
    else
        rounds = recorder.rounds
        s = sum(rounds[1:round])
        e = s + rounds[round + 1]
        return @view(recorder.xs[:, (s + 1):e])
    end
end

struct NoSampleRecorder{T} <: MaybeSampleRecorder{T} end

NoSampleRecorder{T}(::Vector{T}) where {T} = NoSampleRecorder{T}()

record!(::NoSampleRecorder{T}, ::PTChains{N, T, Tr}) where {N, T, Tr} = nothing
