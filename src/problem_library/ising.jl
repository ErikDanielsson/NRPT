struct IsingModel{M, I} <: SamplingProblem
    moment::M       # external field h (Nothing → 0)
    interaction::I  # coupling J       (Nothing → 1)
    N::Int
end

_interaction(::Nothing) = 1.0
_interaction(J::Real) = Float64(J)

_moment(::Nothing) = 0.0
_moment(h::Real) = Float64(h)

function V0(::IsingModel, ::BitArray)
    return 0.0
end

function sample_iid(model::IsingModel)
    return bitrand(model.N, model.N)
end

# Log density: log p(x) ∝ J Σ_{<i,j>} s_i s_j + h Σ_i s_i,  s_i = 2x_i - 1.
# Periodic boundary conditions; each pair counted once (right + down neighbors).
function V1(model::IsingModel, x::BitArray)
    N = model.N
    J = _interaction(model.interaction)
    h = _moment(model.moment)

    log_density = 0.0
    @inbounds for j in 1:N
        jp = j == N ? 1 : j + 1
        for i in 1:N
            ip = i == N ? 1 : i + 1
            si = 2 * x[i, j] - 1
            log_density += J * si * (2 * x[ip, j] - 1)
            log_density += J * si * (2 * x[i, jp] - 1)
        end
    end

    if h != 0.0
        @inbounds for j in 1:N, i in 1:N
            log_density += h * (2 * x[i, j] - 1)
        end
    end

    return log_density
end

struct IsingGibbs <: Explorer end

# Single-site Gibbs sweep over all N² spins in lexicographic order.
# The conditional at site (i,j) given its 4 periodic neighbors and inverse
# temperature β is: P(s_{ij}=+1 | rest) = sigmoid(2β · h_eff),
# where h_eff = J·(sum of neighbor spins) + h.
function step(explorer::IsingGibbs, problem::PathProblem, x::BitArray, β::Float64, lp_buff)
    model = problem.problem::IsingModel
    N = model.N
    J = _interaction(model.interaction)
    h = _moment(model.moment)

    @inbounds for j in 1:N
        jm = j == 1 ? N : j - 1
        jp = j == N ? 1 : j + 1
        for i in 1:N
            im = i == 1 ? N : i - 1
            ip = i == N ? 1 : i + 1

            neighbor_sum = (2 * x[im, j] - 1) + (2 * x[ip, j] - 1) +
                (2 * x[i, jm] - 1) + (2 * x[i, jp] - 1)

            h_eff = β * (J * neighbor_sum + h)
            p_up = 1.0 / (1.0 + exp(-2.0 * h_eff))
            x[i, j] = rand() < p_up
        end
    end
    return x
end
