abstract type BernsteinBasis end

mutable struct ConvexBernstein{M <: Union{Nothing, AbstractMatrix{<:Real}}} <: BernsteinBasis
    order::Int
    l::M
    l_inv::Union{Nothing, Matrix{Float64}}
    eval_matrix::Matrix{Float64}
    cached_schedule::Vector{Float64}
end

function laplacian(n::Int)
    dl = ones(n - 2)
    d = fill(-2.0, n - 1)
    du = ones(n - 2)
    return Tridiagonal(dl, d, du)
end

function ConvexBernstein(order::Int)
    l = order > 1 ? laplacian(order) : nothing
    l_inv = order > 1 ? inv(Matrix(l)) : nothing
    return ConvexBernstein{typeof(l)}(order, l, l_inv, Matrix{Float64}(undef, 0, 0), Float64[])
end

function set_schedule!(basis::ConvexBernstein, schedule::AbstractVector{Float64})
    B = _build_matrix(basis.order, schedule)  # n_sched × (order+1)
    if basis.l_inv !== nothing
        # We transform the evaluation matrix here directly so that when we evaluate the inner point
        # the corresponding polynomial is convex
        A_int = B[:, 2:end-1] * basis.l_inv  # n_sched × (order-1)
        basis.eval_matrix = hcat(B[:, 1] - A_int[:, 1], A_int)
    else
        # order = 1: only one free param c₀; basis_val = c₀ * (1-β_i)
        basis.eval_matrix = B[:, 1:1]
    end
    basis.cached_schedule = collect(schedule)
    return
end

function (basis::ConvexBernstein)(d::AbstractVector, β)
    # This is somewhat ugly, but we do not have access to the index 
    # with the current code structure. find_first gives 
    i = findfirst(==(β), basis.cached_schedule)
    return dot(@view(basis.eval_matrix[i, :]), d)
end

function _build_matrix(order::Int, schedule::AbstractVector{T}) where {T <: Real}
    m = length(schedule)
    b = Matrix{T}(undef, m, order + 1)
    binoms = [T(binomial(order, i)) for i in 0:order]

    t_pows = Vector{T}(undef, order + 1)
    s_pows = Vector{T}(undef, order + 1)

    for (j, t) in enumerate(schedule)
        s = one(T) - t
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

function generate_basis_and_vector(order::Int)
    return ConvexBernstein(order), zeros(order)
end

function eval_schedule_basis(basis::ConvexBernstein, d)
    βs = basis.cached_schedule
    evals = [basis(d, β) for β in βs]
    return βs, evals
end