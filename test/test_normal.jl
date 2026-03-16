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
    @test run.objective_vals[1] > 1000
    @test run.objective_vals[end] < 50
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
    @test run.objective_vals[1] > 1000
    @test run.objective_vals[end] < 50
    @test run.Λ_rej[1] == 9.0
    @test run.Λ_rej[end] < 6
end

@testset "barrier-objective-test" begin
    problem = NormalProblem(0., 1., 1., 1.)
    path = PowerPath(1., AutoForwardDiff())
    ptproblem = PathProblem(problem, path, NormalIIDExplorer())
    optimizer = ProximalStochOptState(
        DoWGState(1e-3, 1e-3),
        ProjectionState(Box(1e-8, 100.))
    )
    n_chains = 3
    n_rounds = 5
    x0 = ones(n_chains)
    init_schedule = collect(range(0, 1, n_chains))
    run = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        warmup=1,
        n_rounds=n_rounds,
        steps_per_round=n -> 5,
        objective=BarrierObjective()
    )
    # Barrier loss is a sum of per-pair rejection rates, bounded in [0, n_chains-1]
    @test all(v -> 0 <= v <= n_chains - 1, run.objective_vals)
    # Barrier should decrease as the path is optimized
    @test run.objective_vals[end] < run.objective_vals[1]
    @test run.Λ_rej[end] < 6
end
