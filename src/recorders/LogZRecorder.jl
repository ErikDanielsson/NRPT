mutable struct LogZRecorder
    schedule_logZsf::Vector{Float64}
    schedule_logZsb::Vector{Float64}
    opt_logZsf::Vector{Float64}
    opt_logZsb::Vector{Float64}
    schedule_round::Int
    opt_round::Int
end

function LogZRecorder(n_schedule_rounds::Int, n_opt_rounds::Int)
    return LogZRecorder(
        Vector{Float64}(undef, n_schedule_rounds),
        Vector{Float64}(undef, n_schedule_rounds),
        Vector{Float64}(undef, n_opt_rounds),
        Vector{Float64}(undef, n_opt_rounds),
        1, 1,
    )
end

function record_schedule!(r::LogZRecorder, logZsf::Float64, logZsb::Float64)
    r.schedule_logZsf[r.schedule_round] = logZsf
    r.schedule_logZsb[r.schedule_round] = logZsb
    r.schedule_round += 1
end

function record_opt!(r::LogZRecorder, logZsf::Float64, logZsb::Float64)
    r.opt_logZsf[r.opt_round] = logZsf
    r.opt_logZsb[r.opt_round] = logZsb
    r.opt_round += 1
end
