struct MHExplorer <: Explorer
    q
end

function step(explorer::MHExplorer, path::Path, x, β) 
    y = rand(explorer.q(x, path, β))
    mh_ratio = (
        log_potential(path, y, β)
        - log_potential(path, x, β)
        - logpdf(explorer.q(x, path, β), y)
        + logpdf(explorer.q(y, path, β), x)
    )
    α = exp(min(0, mh_ratio))
    return !isnan(α) && rand(Bernoulli(α)) ? y : x
end	

function make_q_mala(backend, τ)
    function q_mala(x, path::Path, β)
		annealed_lp = x -> log_potential(path, x, β)
		μ = x + τ * DifferentiationInterface.gradient(annealed_lp, backend, x)
		σ = sqrt(2τ)
		return MvNormal(μ, σ * I)
    end
    return q
end

function make_q_grw(::Type{Vector}, σ=0.1)
    q(T, path::Path, β) = Normal(x, σ * I)
    return q
end

function make_q_grw(::Type{Float64}, σ=0.1)
    q(x, path::Path, β) = Normal(x, σ)
    return q
end

