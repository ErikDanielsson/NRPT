# 1D Cauchy reference Cauchy(0, gamma) → target Cauchy(separation, gamma).
# Narrow Cauchy tails make this a hard transport problem.
function cauchy_slice_sampler(gamma=0.001, separation=100.0; steps=5)
    D0 = Cauchy(0.0, gamma)
    D1 = Cauchy(separation, gamma)
    return (
        GenericDistributionProblem(D0, D1),
        IterExplorer(SliceSampler(), steps)
    )
end
