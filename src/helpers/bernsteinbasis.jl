struct BernsteinBasis
    order::Int
end

(basis::BernsteinBasis)(c::AbstractVector{<:Real}, x) = begin
    n = basis.order
    powers = 1:n-1
    return sum(c[i] * x^i * (1 - x)^(n - i) for i in powers) 
end

function generate_basis_and_vector(order::Int)
    return BernsteinBasis(order), zeros(order - 1)
end