using NRPT
using Test

using Distributions
using DifferentiationInterface
using ForwardDiff

@testset "normal-iid-test" begin
    problem = NormalProblem(0., 1., 100., 1.)
    path = PowerPath(1., AutoForwardDiff())
    ptproblem = PathProblem(problem, path, NormalIIDExplorer())
    optimizer = ProximalStochOptState(
        DoWGState(1e-6, 1e-6),
        ProjectionState(Box(1e-8, 100.))
    )
    n_chains = 10 
    n_rounds = 100
    x0 = ones(n_chains)
    init_schedule = collect(range(0, 1, n_chains))
    run = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 100
    )
    @test run.SKL_ests[1] > 1000
    @test run.SKL_ests[end] < 50
    @test run.Λ_rej[1] == 9.0
    @test run.Λ_rej[end] < 6
end

@testset "normal-slice-test" begin
    problem = NormalProblem(0., 1., 100., 1.)
    path = PowerPath(1., AutoForwardDiff())
    ptproblem = PathProblem(problem, path, IterExplorer(SliceSampler(), 10))
    optimizer = ProximalStochOptState(
        DoWGState(1e-6, 1e-6),
        ProjectionState(Box(1e-8, 100.))
    )
    n_chains = 10 
    n_rounds = 100
    x0 = ones(n_chains)
    init_schedule = collect(range(0, 1, n_chains))
    run = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 100
    )
    @test run.SKL_ests[1] > 1000
    @test run.SKL_ests[end] < 50
    @test run.Λ_rej[1] == 9.0
    @test run.Λ_rej[end] < 6
end