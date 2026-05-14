struct RejectionExplorer{Q} <: IIDExplorer
    q::Q
    lM::Float64
end

function iid_explore(explorer::RejectionExplorer, problem::PathProblem, β::Float64, lp_buff::LP) where {LP <: AbstractVector{Float64}}
    prop_dist = explorer.q(β)
    while true
        y = rand(prop_dist)
        lp_ref = logpdf(prop_dist, y)
        lp = log_potential!(problem, y, β, lp_buff)
        lr = lp - lp_ref
        U = rand()
        if lr >= log(U) + explorer.lM
            return y
        end
    end
    return
end
