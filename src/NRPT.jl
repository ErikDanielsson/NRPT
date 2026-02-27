module NRPT

using Distributions, Random, LinearAlgebra, Interpolations, ProgressMeter, ADTypes
import DifferentiationInterface

include("helpers.jl")
include("log_potential/sampling_problems.jl")
include("log_potential/paths.jl")

include("evidence/stepping_stone.jl")
include("explorers/explorers.jl")
include("explorers/mh_kernels.jl")
include("explorers/slice_sampler.jl")

include("log_potential/path_problem.jl")

include("opt/stoch_opt.jl")
include("opt/proximal_operators.jl")
include("opt/optimizers.jl")
include("opt/path_optimization.jl")

include("stats/barriers.jl")
include("stats/stats.jl")

include("DEO.jl")
include("nrpt.jl")

# Explorers
export MHExplorer, SliceSampler, IterExplorer
export make_q_mala, make_q_grw

# Problems and choices of paths
export SamplingProblem, PathProblem, linear_path, power_path, q_path, spline_path
export linear_spline, theta_to_eta

# Statistics
export round_trip_rate, count_chain_round_trips, count_round_trips_per_round

# Optimizaton
export ProximalStochOptState, NoOptState
export ProximalState, ProjectionState, Box, project
export SGDState, DoGState, DoWGState

# Main algorithm
export nrpt, optimized_nrpt


end