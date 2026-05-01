using LogExpFunctions


logmeanexp(X; dims=1) = logsumexp(X; dims=dims) .- log(size(X, dims))

function logweightaddexp(a, x, b, y)
    m = max(x, y)
    return m + log(max(a * exp(x - m) + b * exp(y - m), 0.0))
end

norm2(x::T) where {T<:Real} = x^2 
norm2(x::T) where {T<:AbstractArray}= sum(abs2, x)

function logcummeanexp(X)
    n = length(X)
    result = Vector{Float64}(undef, n)
    for i in 1:n
        # Compute log(mean(exp(X[1:i]))) = logsumexp(X[1:i]) - log(i)
        logmean_i = logsumexp(X[1:i]) .- log(i)
        result[i] = logmean_i
    end
    
    return result
end

function nan_grad(f::T) where {T <: Real}
    return isnan(f)
end

function nan_grad(f::T) where {T <: AbstractArray}
    return any(isnan, f)
end

softmax_(x) = exp.(x .- logsumexp_(x))
logsumexp_(x) = LogExpFunctions.logsumexp(x)

function logsumexp_(x::Vector{ForwardDiff.Dual{T,V,N}}) where {T, V, N}
    x_val = ForwardDiff.value.(x)
    m = maximum(x_val)
    S = zero(V)
    acc = zero(ForwardDiff.Partials{N,V})
    @inbounds for i in eachindex(x)
        e  = exp(x_val[i] - m)
        S += e
        acc += e * ForwardDiff.partials(x[i])
    end
    val = log(S) + m
    p   = acc / S   # softmax weighting, done once
    return ForwardDiff.Dual{T,V,N}(val, p)
end

function LogExpFunctions.logsumexp(
    u::AbstractVector{ForwardDiff.Dual{T2, ForwardDiff.Dual{T1,V,N1}, N2}},
) where {T1, T2, V, N1, N2}
    n = length(u)

    # Only unavoidable length-n buffers
    x_val = [ForwardDiff.value(ForwardDiff.value(u[i])) for i in 1:n]
    val   = LogExpFunctions.logsumexp(x_val)
    s     = exp.(x_val .- val)                    # softmax; sum(s) ≈ 1

    # Accumulators — all small (length N1, N2, or N1*N2)
    g_P1     = zero(ForwardDiff.Partials{N1,V})           # s' * P1

    # Pass 1: g_P1, g_P2_val, and cross[k] for each k
    cross = ntuple(_ -> zero(ForwardDiff.Partials{N1,V}), N2)
    g_P2_val_acc = zero.(ntuple(_ -> zero(V), N2))
    # Julia tuples are immutable; rebuild each iteration. For small N2 the compiler unrolls.
    @inbounds for i in 1:n
        outer_i = u[i]
        inner_i = ForwardDiff.value(outer_i)               # Dual{T1,V,N1}
        p1_i    = ForwardDiff.partials(inner_i)            # Partials{N1,V}
        g_P1   += s[i] * p1_i

        cross = ntuple(N2) do k
            d = ForwardDiff.partials(outer_i, k)           # Dual{T1,V,N1}
            cross[k] + s[i] * ForwardDiff.partials(d)
        end
        g_P2_val_acc = ntuple(N2) do k
            d = ForwardDiff.partials(outer_i, k)
            g_P2_val_acc[k] + s[i] * ForwardDiff.value(d)
        end
    end

    # Pass 2: P1tHP2[k] = Σ_i p1_i * s[i] * (P2_val[i,k] - g_P2_val_acc[k])
    P1tHP2 = ntuple(_ -> zero(ForwardDiff.Partials{N1,V}), N2)
    @inbounds for i in 1:n
        outer_i = u[i]
        p1_i    = ForwardDiff.partials(ForwardDiff.value(outer_i))
        P1tHP2  = ntuple(N2) do k
            d   = ForwardDiff.partials(outer_i, k)
            wk  = s[i] * (ForwardDiff.value(d) - g_P2_val_acc[k])
            P1tHP2[k] + wk * p1_i
        end
    end

    # Pack result
    inner_val = ForwardDiff.Dual{T1,V,N1}(val, g_P1)

    outer_parts = ntuple(N2) do k
        ForwardDiff.Dual{T1,V,N1}(g_P2_val_acc[k], cross[k] + P1tHP2[k])
    end

    return ForwardDiff.Dual{T2, ForwardDiff.Dual{T1,V,N1}, N2}(
        inner_val,
        ForwardDiff.Partials{N2, ForwardDiff.Dual{T1,V,N1}}(outer_parts),
    )
end