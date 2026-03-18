using LogExpFunctions


logmeanexp(X; dims=1) = logsumexp(X; dims=dims) .- log(size(X, dims))

function logweightaddexp(a, x, b, y)
    m = max(x, y)
    return m + log(a * exp(x - m) + b * exp(y - m))
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

