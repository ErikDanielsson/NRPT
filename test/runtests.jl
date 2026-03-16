using NRPT
using Test

using Distributions
using DifferentiationInterface
using ForwardDiff

@testset "NRPT.jl" begin
    include("test_normal.jl")    
end
