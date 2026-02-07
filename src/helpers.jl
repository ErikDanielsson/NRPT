using LogExpFunctions

logmeanexp(X; dims=1) = logsumexp(X; dims=dims) .- log(size(X, dims))