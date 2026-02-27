abstract type Explorer end

struct IterExplorer{E <: Explorer} <: Explorer
    explorer::E
    n::Int
end

function step(explorer::IterExplorer, path::Path, x, β)
    for _ in 1:explorer.n
        x = step(explorer.explorer, path, x, β)
    end
    return x
end