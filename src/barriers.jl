# Use the pure rejection probabilities 
function λ_est_simple(r::Matrix{Float64})
	return exp.(logmeanexp(log.(r); dims=1))
end

# Use both the rejection and acceptance probabilities
# This will penalize difficult regions more
function λ_est_accept(r::Matrix{Float64})
	return exp.(logmeanexp(log.(r) - log.(1 .- r)); dims=1)
end 

function make_schedule(r, schedule; use_accept=false)
	λ = use_accept ? λ_est_complicated(r) : λ_est_simple(r)
	Λβ = [0; cumsum(λ)]
	norm_Λβ = Λβ ./ last(Λβ)
	if length(unique(norm_Λβ)) != length(norm_Λβ)
		λ_ = λ .+  1e-6
		Λβ_ = [0; cumsum(λ_)]
		norm_Λβ = Λβ_ ./ last(Λβ_)
	end
	uniform = range(0., 1., length(schedule))
	generator = interpolate(norm_Λβ, schedule, FritschCarlsonMonotonicInterpolation())
	
	return barrier, [0.0; generator.(uniform[2:end-1]); 1.0]
end
