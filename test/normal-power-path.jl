@testset "normal-iid-test" begin
    problem = NormalProblem(0.0, 1.0, 100.0, 1.0)
    t0 = 1.0
    path = PowerPath(t0, AutoForwardDiff())
    ptproblem = PathProblem(problem, path, NormalIIDExplorer())
    optimizer = ProximalStochOptState(
        DoWGState(1.0e-3, 1.0e-3),
        ProjectionState(Box(1.0e-10, 1.0e10))
    )
    n_chains = 10
    n_rounds = 1000
    x0 = ones(n_chains)
    init_schedule = collect(range(0, 1, n_chains))
    run2 = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        n_rounds = n_rounds,
        steps_per_round = n -> 100
    )
    @test run.loss_recorder.skl[1] > 1000
    @test run.loss_recorder.skl[end] < 50
    @test run.schedule_recorder.Λ_rej[1] == 9.0
    @test run.schedule_recorder.Λ_rej[end] < 6
end

@testset "normal-slice-test" begin
    problem = NormalProblem(0.0, 1.0, 100.0, 1.0)
    path = PowerPath(1.0, AutoForwardDiff())
    ptproblem = PathProblem(problem, path, IterExplorer(SliceSampler(), 10))
    optimizer = ProximalStochOptState(
        DoWGState(1.0e-6, 1.0e-6),
        ProjectionState(Box(1.0e-8, 100.0))
    )
    n_chains = 10
    n_rounds = 100
    x0 = ones(n_chains)
    init_schedule = collect(range(0, 1, n_chains))
    run = optimized_nrpt(
        x0, init_schedule, ptproblem, optimizer;
        n_rounds = n_rounds,
        steps_per_round = n -> 100
    )
    @test run.loss_recorder.skl[1] > 1000
    @test run.loss_recorder.skl[end] < 50
    @test run.schedule_recorder.Λ_rej[1] == 9.0
    @test run.schedule_recorder.Λ_rej[end] < 6
end
