struct IterExplorer{E <: Explorer} <: Explorer
    explorer::E
    n::Int
end

function step(explorer::IterExplorer, problem::PathProblem, x::T, β, lp_buff::LP) where {T <: AbstractVector, LP <: AbstractVector{Float64}}
    for _ in 1:explorer.n
        x = step(explorer.explorer, problem, x, β, lp_buff)
    end
    return x
end