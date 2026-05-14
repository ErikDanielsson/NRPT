using NRPT
using Test

using Distributions
using DifferentiationInterface
using ForwardDiff

@testset "NRPT.jl" begin
    # include("test_normal.jl")
    # include("test_gbm_path_normal.jl")
    include("test_gbm_path_bimodal.jl")
    # include("test_gbm_path_uniform.jl")
end
