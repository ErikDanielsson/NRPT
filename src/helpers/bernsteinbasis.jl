abstract type BernsteinBasis{C <: Val} end

struct ConvexBernstein{C <: Val, V <: AbstractVector{<:Real}, M <: Union{Nothing, AbstractMatrix{<:Real}}} <: BernsteinBasis{C}
    order::Int
    binom_cache::V
    l::M
end

function laplacian(n::Int)
    dl = ones(n - 2)
    d = fill(-2.0, n - 1)
    du = ones(n - 2)
    return Tridiagonal(dl, d, du)
end

function ConvexBernstein(order::Int, endpoint::Bool)
    binom_cache = SVector{order + 1}([Float64(binomial(order, i)) for i in 0:order])
    l = order > 1 ? laplacian(order) : nothing
    return ConvexBernstein{Val{endpoint}, typeof(binom_cache), typeof(l)}(order, binom_cache, l)
end

# From the central diff param. compute the actual parameters of the polynomial
function cvx_coeffs(basis::BernsteinBasis, c0, cn, d)
    l = basis.l
    rhs = copy(d)
    rhs[1] -= c0
    rhs[end] -= cn
    c_interior = l \ rhs
    return c_interior
end

function (basis::ConvexBernstein{Val{true}, <:AbstractVector, <:AbstractMatrix})(d::V, β) where {V <: AbstractVector}
    T = eltype(d)
    inner_coeffs = cvx_coeffs(basis, zero(T), zero(T), d)
    return _eval_bernstein(basis.order, basis.binom_cache, [zero(T); inner_coeffs; zero(T)], β)
end

function (basis::ConvexBernstein{Val{false}, <:AbstractVector, <:AbstractMatrix})(d::V, β) where {V <: AbstractVector}
    T = eltype(d)
    inner_coeffs = cvx_coeffs(basis, d[1], zero(T), @view(d[2:end]))
    return _eval_bernstein(basis.order, basis.binom_cache, [d[1]; inner_coeffs; zero(T)], β)
end

function (basis::ConvexBernstein{Val{false}, <:AbstractVector, Nothing})(d::V, β) where {V <: AbstractVector}
    T = eltype(d)
    return _eval_bernstein(basis.order, basis.binom_cache, [d[1], zero(T)], β)
end


function _eval_bernstein(n, binom_cache, c, β)
    T = promote_type(eltype(c), typeof(β))
    s = β
    t = one(T) - β
    v = zero(T)
    # build (1-β)^(n-i) on the fly via running product going the other way
    # — but easier: just compute both power arrays in MVectors
    s_pows = MVector{n + 1, T}(undef)   # needs n known statically
    t_pows = MVector{n + 1, T}(undef)
    s_pows[1] = one(T); t_pows[1] = one(T)
    @inbounds for i in 1:n
        s_pows[i + 1] = s_pows[i] * s
        t_pows[i + 1] = t_pows[i] * t
    end
    @inbounds for i in 1:(n + 1)
        v += c[i] * binom_cache[i] * s_pows[i] * t_pows[n + 2 - i]
    end
    return v
end

function generate_basis_and_vector(order::Int, endpoint::Bool)
    return ConvexBernstein(order, endpoint), zeros(order - Int(endpoint))
end

# For a fixed schedule, evaluate the Bernstein basis for different parameter values
mutable struct BernsteinBasisInplace{C <: Val, A <: AbstractMatrix{<:Real}} <: BernsteinBasis{C}
    order::Int
    b::A
end

function BernsteinBasisInplace(order, endpoint, schedule)
    b = _build_matrix(order, schedule)
    return BernsteinBasisInplace{Val{endpoint}, typeof(b)}(order, b)
end

function _build_matrix(order::Int, schedule::AbstractVector{T}) where {T <: Real}
    m = length(schedule)
    b = Matrix{T}(undef, m, order + 1)
    binoms = [T(binomial(order, i)) for i in 0:order]   # precompute once

    t_pows = Vector{T}(undef, order + 1)
    s_pows = Vector{T}(undef, order + 1)

    for (j, t) in enumerate(schedule)
        s = one(T) - t
        # iterative powers — no repeated exponentiation
        t_pows[1] = one(T); s_pows[1] = one(T)
        @simd for i in 1:order
            t_pows[i + 1] = t_pows[i] * t
            s_pows[i + 1] = s_pows[i] * s
        end
        @inbounds for i in 0:order
            b[j, i + 1] = binoms[i + 1] * t_pows[i + 1] * s_pows[order - i + 1]
        end
    end
    return b
end

function (basis::BernsteinBasisInplace{Val{false}, A})(c::AbstractVector{T}) where {A, T <: Real}
    return basis.b * [c; one(T)]
end

function (basis::BernsteinBasisInplace{Val{true}, A})(c::AbstractVector{T}) where {A, T <: Real}
    return basis.b * [one(T); c; one(T)]
end

# function convexity_violation(::BernsteinBasis{Val{endpoint}}, c) where {endpoint}
#     v = endpoint ? [1.0; c; 1.0] : [c; 1.0]
#     n = length(v)
#     Δs = Vector{Float64}(undef, n - 2)
#     for i in 2:n-1
#         Δs[i - 1] = v[i - 1] - 2v[i] + v[i + 1]
#     end
#     return Δs
# end
