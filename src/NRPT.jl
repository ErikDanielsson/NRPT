module NRPT

using Distributions, Random, LinearAlgebra, Interpolations, ProgressMeter, ADTypes
import DifferentiationInterface

include("helpers/misc.jl")
include("helpers/polydecayaverager.jl")
export cumavg
# Path problem 
include("log_potential/path_problem.jl")
export PathProblem, run_single_chain

# Sampling problems
include("log_potential/sampling_problems.jl")
export PosteriorProblem, GenericDistributionProblem
export NormalProblem, exponents_to_params

# Path types
include("log_potential/paths.jl")
export get_exponents, extract_params 
include("log_potential/paths/LinearPath.jl")
export LinearPath
include("log_potential/paths/PowerPath.jl")
export PowerPath
include("log_potential/paths/QPath.jl")
export QPath

include("log_potential/paths/spline-helpers.jl")
include("log_potential/paths/PaperSplinePath.jl")
export PaperSplinePath
include("log_potential/paths/SplinePath.jl")
export SplinePath
include("log_potential/paths/SingleSplinePath.jl")
export SingleSplinePath

# Explorers
include("explorers/iterexplorer.jl")
export IterExplorer
include("explorers/mh_kernels.jl")
export MHExplorer, make_q_mala, make_q_grw
include("explorers/slice_sampler.jl")
export SliceSampler
include("explorers/iid_explorer.jl")
export NormalIIDExplorer

include("Chain.jl")
include("PTChains.jl")

# Optimization
include("opt/stoch_opt.jl")
include("opt/SGD_variants/SGD.jl")
export SGDState
include("opt/SGD_variants/DoG.jl")
export DoGState
include("opt/SGD_variants/DoWG.jl")
export DoWGState
include("opt/SGD_variants/Adam.jl")
export AdamState
include("opt/SGD_variants/Adagrad.jl")
export AdagradState
include("opt/SGD_variants/ScaledAdagrad.jl")
export ScaledAdagradState
include("opt/proximal_operators.jl")
export ProximalState, ProjectionState, Box, project
include("opt/optimizers.jl")
export ProximalStochOptState, NoOptState
include("opt/objectives/SKL.jl")
include("opt/objectives/rejection-estimator.jl")
include("opt/objectives/objectives.jl")
export SKLObjective, BarrierObjective
include("opt/path_optimization.jl")

# Statistics
include("stats/barriers.jl")
include("stats/stepping_stone.jl")
include("stats/stats.jl")
export round_trip_rate, count_chain_round_trips, count_round_trips_per_round

# Main algorithm
include("DEO.jl")
include("nrpt.jl")
export nrpt, optimized_nrpt


end