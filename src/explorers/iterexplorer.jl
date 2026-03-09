struct IterExplorer{E <: Explorer} <: Explorer
    explorer::E
    n::Int
end

function step(explorer::IterExplorer, problem::PathProblem, x, β)
    for _ in 1:explorer.n
        x = step(explorer.explorer, problem, x, β)
    end
    return x
end