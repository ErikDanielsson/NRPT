module NRPT

using Distributions, Random, LinearAlgebra, Interpolations, ProgressMeter, ADTypes
import DifferentiationInterface

include("helpers.jl")
include("problems.jl")
include("evidence/stepping_stone.jl")
include("explorers/explorers.jl")
include("explorers/mh_kernels.jl")
include("explorers/slice_sampler.jl")

include("path-problem.jl")
include("barriers.jl")

include("opt/stoch-opt.jl")
include("opt/proximal-operators.jl")
include("opt/optimizers.jl")

include("stats.jl")
include("path_adaptation.jl")
include("DEO.jl")

include("nrpt.jl")

export make_q_mala, make_q_grw
export MHExplorer, SliceSampler, IterExplorer
export nrpt, optimized_nrpt
export SamplingProblem, PathProblem, linear_path, power_path, q_path, spline_path
export linear_spline, theta_to_eta
export round_trip_rate, count_chain_round_trips, count_round_trips_per_round
export ProximalStochOptState, NoOptState
export ProximalState, ProjectionState, Box, project
export SGDState, DoGState, DoWGState


end