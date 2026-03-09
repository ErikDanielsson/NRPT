mutable struct SliceSampler <: Explorer
    w::Float64
    p::Int
end

SliceSampler() = SliceSampler(10., 3)

function step(explorer::SliceSampler, problem::PathProblem, x::Float64, β)
    # Get the current level (z = log(f(x)))
    z = log_potential(problem, x, β) - rand(Exponential())
    U = rand()
    L = x - U * explorer.w
    R = L + explorer.w
    K = explorer.p
    while (K > 0
        && (
            z < log_potential(problem, L, β) || z < log_potential(problem, R, β)
        )
    )
        V = rand()
        if V < 0.5
            L = 2L - R
        else
            R = 2R - L
        end
        K -= 1
    end
    return shrink(explorer, problem, L, R, z, x, β)
end

function shrink(explorer::SliceSampler, problem::PathProblem, L::Float64, R::Float64, z::Float64, x::Float64, β)
    L_bar = L
    R_bar = R
    U = rand()
    y = L_bar + U * (R_bar - L_bar)
    while true
        if (
            z <= log_potential(problem, y, β)
            && accept(explorer, problem, L, R, z, x, y, β)
        )
            break
        end
        if y < x
            L_bar = y
        else
            R_bar = y
        end
        U = rand()
        y = L_bar + U * (R_bar - L_bar)
    end
    return y
end

function accept(explorer::SliceSampler, problem::PathProblem, L::Float64, R::Float64, z::Float64, x::Float64, y::Float64, β)
    L_hat = L 
    R_hat = R 
    D = false
    while (R_hat - L_hat) > 1.1 * explorer.w
        M = (R_hat + L_hat) / 2.0
        if (x < M && y >= M) || (x >= M && y < M)
            D = true
        end
        if y < M 
            R_hat = M
        else
            L_hat = M
        end
        if (D
            && z >= log_potential(problem, L_hat, β)
            && z >= log_potential(problem, R_hat, β)
        )
            return false
        end
    end
    return true
end