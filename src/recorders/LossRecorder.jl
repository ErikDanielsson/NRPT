abstract type LossRecorder end

mutable struct SKLRecorder <: LossRecorder
    Λ_opt_round::Vector{Float64}
    skl::Vector{Float64}
    opt_round::Int
end

function SKLRecorder(n_opt_rounds::Int)
    return SKLRecorder(
        Vector{Float64}(undef, n_opt_rounds),
        Vector{Float64}(undef, n_opt_rounds),
        1,
    )
end

function record!(recorder::SKLRecorder, Λ::Float64, skl::Float64)
    recorder.Λ_opt_round[recorder.opt_round] = Λ
    recorder.skl[recorder.opt_round] = skl
    recorder.opt_round += 1
end
