abstract type LossRecorder end

struct SKLRecorder{A <: Union{Int, Nothing}} <: LossRecorder
    Λ_schedule_round::Vector{Float64}
    Λ_opt_round::Vector{Float64}
    skl::Vector{Float64}
    adaptation_start::A
    schedule_round::Int
    opt_round::Int
end

function SKLRecorder(n_rounds, adaptation_start::A) where {A <: Union{Int, Nothing}}
    Λ_schedule_round = Vector{Float64}(undef, n_rounds)
    Λ_opt_round = Vector{Float64}(undef, n_rounds)
    skl = Vector{Float64}(undef, n_rounds)
    return SKLRecorder{A}(Λ_schedule_round, Λ_opt_round, skl, adaptation_start, 1, 1)
end

function record_schedule_Λ!(recorder::SKLRecorder, Λ::Float64)
    recorder.Λ_schedule_round[recorder.schedule_round] = Λ
    recorder.schedule_round += 1
end

function record_opt!(recorder::SKLRecorder, Λ::Float64, skl::Float64)
    recorder.Λ_opt_round[recorder.opt_round] = Λ
    recorder.skl[recorder.opt_round] = skl
    recorder.opt_round += 1
end
