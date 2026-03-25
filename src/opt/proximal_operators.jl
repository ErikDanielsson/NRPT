abstract type ProximalState end

abstract type ProjectionSet end

struct NoProx <: ProximalState end

step!(x, ::NoProx) = x

struct ProjectionState{R <: ProjectionSet} <: ProximalState
    set::R
end

function step!(x, state::ProjectionState) 
    return project(x, state.set)
end

struct Box{T} <: ProjectionSet
    lb::T
    ub::T
end

project(x::Vector{Float64}, box::Box{Vector{Float64}}) = max.(min.(x, box.ub), box.lb)
project(x::Float64, box::Box{Float64}) = max(min(x, box.ub), box.lb)

struct HalfSpace{T} <: ProjectionSet
    n::T
    b::Float64
end
function project(x::Vector{Float64}, hs::HalfSpace{Vector{Float64}}) 
    d = dot(hs.n, x)
    if d <= hs.b
        return x
    else
        return x * (1 - (d - hs.b) / norm2(x))
    end
end
function project(x::Float64, hs::HalfSpace{Float64}) 
    d = hs.n * x 
    if d <= hs.b
        return x
    else
        return x * (1 - (d - hs.b) / x^2)
    end
end

struct SumConstraint <: ProjectionSet
    b::Float64
end
function project(x::Vector{Float64}, sc::SumConstraint) 
    d = sum(x[2:end])
    if d <= sc.b
        return x
    else
        return [x[1]; x[2:end] * (1 - (d - sc.b) / norm2(x[2:end]))]
    end
end

# struct MonotoneSequence <: ProjectionSet
# end

# project(x::Matrix{Float64}, box::Box{Vector{Float64}}) = max.(min.(x, box.ub), box.lb)
# project(x::Float64, box::Box{Float64}) = max(min(x, box.ub), box.lb)

