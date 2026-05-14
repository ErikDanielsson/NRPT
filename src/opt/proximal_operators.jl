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

struct LowerBound{T} <: ProximalState
    lb::T
end

step!(x, lb::LowerBound) = max.(lb.lb, x)

struct Box{T} <: ProjectionSet
    lb::T
    ub::T
end

project(x::Vector{Float64}, box::Box{Vector{Float64}}) = max.(min.(x, box.ub), box.lb)
project(x::Float64, box::Box{Float64}) = max(min(x, box.ub), box.lb)