struct Indices
    σ::Vector{Int}
    σ_inv::Vector{Int}
end

Indices(n::Int) = Indices(1:n, 1:n)
copy(is::Indices) = Indices(Base.copy(is.σ), Base.copy(is.σ_inv))

function swap(is::Indices, i, j)
	new_is = copy(is)
	new_is.σ[i] = is.σ[j]
	new_is.σ[j] = is.σ[i]
	new_is.σ_inv[is.σ[i]] = j
	new_is.σ_inv[is.σ[j]] = i
	return new_is
end