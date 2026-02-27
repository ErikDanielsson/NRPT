mutable struct SliceSampler <: Explorer
    w::Float64
    p::Int
end


function step(kernel::SliceSampler, path::Path, x::Float64, β)
    # Get the current level (z = log(f(x)))
    z = log_potential(path, x, β) - rand(Exponential())
    U = rand()
    L = x - U * kernel.w
    R = L + kernel.w
    K = kernel.p
    while (K > 0
        && (
            z < log_potential(path, L, β) || z < log_potential(path, R, β)
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
    # println("L: $L, R: $R")
    return shrink(kernel, L, R, z, path, x, β)
end

function shrink(kernel::SliceSampler, L::Float64, R::Float64, z::Float64, path::Path, x::Float64, β)
    L_bar = L
    R_bar = R
    U = rand()
    y = L_bar + U * (R_bar - L_bar)
    while true
        # println("Shrinking ($L_bar  - $R_bar): $(y) - $x, Log potential: $(log_potential(path, y, β)) - $z")
        if (
            z <= log_potential(path, y, β)
            && accept(kernel, L, R, z, path, x, y, β)
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

function accept(kernel::SliceSampler, L, R, z, path::Path, x, y, β; ϵ=0.1)
    L_hat = L 
    R_hat = R 
    D = false
    while (R_hat - L_hat) > 1.1 * kernel.w
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
            && z >= log_potential(path, L_hat, β)
            && z >= log_potential(path, R_hat, β)
        )
            return false
        end
    end
    return true
end