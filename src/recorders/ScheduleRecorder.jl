mutable struct ScheduleRecorder
    schedules::Matrix{Float64}
    barriers::Vector{Any}
    Λ_rej::Vector{Float64}
    Λ_acc::Vector{Float64}
    round::Int
end

function ScheduleRecorder(n_chains::Int, n_rounds::Int, initial_schedule::Vector{Float64})
    schedules = Matrix{Float64}(undef, n_chains, n_rounds + 1)
    schedules[:, 1] = initial_schedule
    return ScheduleRecorder(
        schedules,
        Vector{Any}(undef, n_rounds),
        Vector{Float64}(undef, n_rounds),
        Vector{Float64}(undef, n_rounds),
        1,
    )
end

function record!(r::ScheduleRecorder, schedule::Vector{Float64}, Λ_β, Λ_rej::Float64, Λ_acc::Float64)
    r.schedules[:, r.round + 1] = schedule
    r.barriers[r.round] = Λ_β
    r.Λ_rej[r.round] = Λ_rej
    r.Λ_acc[r.round] = Λ_acc
    r.round += 1
end
