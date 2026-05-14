mutable struct PolynomialDecayAverager{T, P <: Union{Nothing, Vector{T}}}
    x_bar_t::T
    xs::P
    gamma::Float64
end

PolynomialDecayAverager(gamma; save_iterates = false) =
    PolynomialDecayAverager(nothing, save_iterates ? [nothing] : nothing, gamma)

function init_averager(x0, averager::PolynomialDecayAverager{Nothing, Vector{Nothing}})
    return PolynomialDecayAverager(x0, [x0], averager.gamma)
end

function init_averager(x0, averager::PolynomialDecayAverager{Nothing, Nothing})
    return PolynomialDecayAverager(x0, nothing, averager.gamma)
end

function next_iterate(x::T, x_bar_t::T, t, gamma) where {T}
    w_t = (1 + gamma) / (t + gamma)
    return (1 - w_t) * x_bar_t + w_t * x
end

function update!(x::T, t, averager::PolynomialDecayAverager{T, Vector{T}}) where {T}
    new_x_bar_t = next_iterate(x, averager.x_bar_t, t, averager.gamma)
    averager.x_bar_t = new_x_bar_t
    return push!(averager.xs, new_x_bar_t)
end

function update!(x::T, t, averager::PolynomialDecayAverager{T, Nothing}) where {T}
    return averager.x_bar_t = next_iterate(x, averager.x_bar_t, t, averager.gamma)
end

function get_average(averager::PolynomialDecayAverager{T, P}) where {T, P}
    return averager.x_bar_t
end

function get_samples(averager::PolynomialDecayAverager{T, Vector{T}}) where {T}
    return averager.xs
end

# Cumulative average of a vector with polynomial decay, returning all intermediate averages
function cumavg(xs::AbstractVector{T}, gamma) where {T}
    averager = init_averager(xs[1], PolynomialDecayAverager(gamma; save_iterates = true))
    for t in 2:length(xs)
        update!(xs[t], t, averager)
    end
    return get_samples(averager)
end
