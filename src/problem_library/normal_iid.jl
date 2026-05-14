# 1D Normal reference N(mu0, sigma0) → target N(mu1, sigma1).
# Uses NormalIIDExplorer, which exploits the Gaussian structure for exact IID sampling.
function normal_iid(; mu0 = -1.0, sigma0 = 0.01, mu1 = 1.0, sigma1 = 0.01)
    return (
        NormalProblem(mu0, sigma0, mu1, sigma1),
        NormalIIDExplorer(),
    )
end
