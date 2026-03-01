function stepping_stone(lps_forward::Matrix{Float64})
    l = logmeanexp(lps_forward; dims=2)
    return sum(l)
end