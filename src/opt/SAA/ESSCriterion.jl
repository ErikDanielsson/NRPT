abstract type ESSCriterion end

struct FixedrESSCriterion <: ESSCriterion
    δ::Float64
end

get_lb(crit::FixedrESSCriterion, ::Int) = crit.δ

struct DecayrESSCriterion <: ESSCriterion
    c::Float64
    rate::Float64
end

function get_lb(crit::DecayrESSCriterion, n_samples::Int)
    return crit.c / n_samples^crit.rate
end