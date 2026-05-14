function save_rejections(save_r, new_r; save = false)
    return save ? hcat(save_r, new_r) : save_r
end

# Use the pure rejection probabilities
function λ_est_simple(r::Matrix{Float64})
    return vec(exp.(logmeanexp(log.(r); dims = 2)))
end

# Use both the rejection and acceptance probabilities
# This will penalize difficult regions more
function λ_est_accept(r::Matrix{Float64})
    p(r; ϵ = 1) = (1 + ϵ) / (1 + ϵ - r)
    return vec(exp.(logmeanexp(log.(r) + log.(p.(r)); dims = 2)))
end

function compute_Λ(r, schedule; use_accept = false)
    λ = use_accept ? λ_est_accept(r) : λ_est_simple(r)
    return sum(λ)
end

function proj_interval(x; a = 0.0, b = 1.0)
    return min(b, max(a, x))
end

function make_schedule(r, schedule; use_accept = false, min_incr = 1.0e-15)
    λ = use_accept ? λ_est_accept(r) : λ_est_simple(r)
    Λβ = sort([0; cumsum(λ)])
    norm_Λβ = Λβ ./ last(Λβ)
    if length(unique(norm_Λβ)) != length(norm_Λβ)
        λ_ = λ .+ 1.0e-6
        Λβ = sort([0; cumsum(λ_)])
        norm_Λβ = Λβ ./ last(Λβ)
        if length(unique(norm_Λβ)) != length(norm_Λβ)
            barrier = interpolate(schedule, Λβ, FritschCarlsonMonotonicInterpolation())
            return barrier, schedule
        end
    end
    uniform = range(0.0, 1.0, length(schedule))
    generator = interpolate(norm_Λβ, schedule, FritschCarlsonMonotonicInterpolation())
    # println(Λβ)
    try
        barrier = interpolate(schedule, Λβ, FritschCarlsonMonotonicInterpolation())
    catch e
        println(e)
        println(schedule)
        println(Λβ)
        throw(e)
    end
    schedule = [0.0; generator.(uniform[2:(end - 1)]); 1.0]
    if min_incr > 0.0
        for i in 2:(length(schedule) - 1)
            if abs(schedule[i] - schedule[i - 1]) < min_incr
                schedule[i] = proj_interval(schedule[i - 1] + min_incr)
            end
        end
    end
    # @info "Schedule = $schedule"
    return barrier, schedule
end
