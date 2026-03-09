struct MHExplorer <: Explorer
    q
end

function step(explorer::MHExplorer, problem::PathProblem, x, β) 
    y = rand(explorer.q(x, path, β))
    mh_ratio = (
        log_potential(problem, y, β)
        - log_potential(problem, x, β)
        - logpdf(explorer.q(x, problem, β), y)
        + logpdf(explorer.q(y, problem, β), x)
    )
    α = exp(min(0, mh_ratio))
    return !isnan(α) && rand(Bernoulli(α)) ? y : x
end	

function make_q_mala(backend, τ)
    function q_mala(x, problem::PathProblem, β)
		annealed_lp = x -> log_potential(problem, x, β)
		μ = x + τ * DifferentiationInterface.gradient(annealed_lp, backend, x)
		σ = sqrt(2τ)
		return MvNormal(μ, σ * I)
    end
    return q
end

function make_q_grw(::Type{Vector}, σ=0.1)
    q(x, problem::PathProblem, β) = Normal(x, σ * I)
    return q
end

function make_q_grw(::Type{Float64}, σ=0.1)
    q(x, problem::PathProblem, β) = Normal(x, σ)
    return q
end

