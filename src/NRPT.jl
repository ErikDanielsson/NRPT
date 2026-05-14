module NRPT

using Distributions, Random, LinearAlgebra, Interpolations
using ProgressMeter, ADTypes, DelimitedFiles, OhMyThreads
using StaticArrays
import DifferentiationInterface, BSplineKit, ForwardDiff

include("helpers/misc.jl")
include("helpers/bernsteinbasis.jl")
export ConvexBernstein
# Path problem
include("log_potential/path_problem.jl")
export PathProblem, run_single_chain

# Sampling problems
include("log_potential/sampling_problems.jl")
export PosteriorProblem, GenericDistributionProblem, MvUnivariate
export NormalProblem, exponents_to_params
include("log_potential/gaussian_base_measure.jl")
export GBM, GBMProblem, GaussianGBM, UniformGBM, BoundedUniformGBM, Likelihood, loglik, GaussianLikelihood

# Path types
include("log_potential/paths.jl")
export get_exponents, extract_params
include("log_potential/paths/LinearPath.jl")
export LinearPath
include("log_potential/paths/QPath.jl")
export QPath
include("log_potential/paths/SymmetricPerturbed.jl")
export SymmetricPerturbed

include("log_potential/paths/GBMPath.jl")
export ScalingGBMPath


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
include("opt/proximal_operators.jl")
export ProximalState, NoProx, ProjectionState, LowerBound, Box
include("opt/optimizers.jl")
export ProximalStochOptState, NoOptState

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



include("opt/objectives.jl")
include("opt/SAA/SNISSKLLoss.jl")
include("opt/SAA/ESSCriterion.jl")
export ESSCriterion, FixedrESSCriterion, DecayrESSCriterion
include("opt/SAA/modified-newtons-method.jl")
export NewtonTrustRegionState
include("opt/SAA/ISSAA.jl")

include("opt/SA/SGD-SKL.jl")
include("opt/SA/rejection-estimator.jl")
export SKLObjective, BarrierObjective, TrustRegionState, NewtonTrustRegionBarrierState
include("opt/path_optimization.jl")

# Recorders
include("recorders/IndexProcess.jl")
export IndexProcess, get_round_index_proc, round_trip_rate
include("recorders/SampleRecorder.jl")
export SampleRecorder, get_round_samples
include("recorders/ScheduleRecorder.jl")
export ScheduleRecorder, get_barriers, get_schedules, get_Λ_rej, get_Λ_acc
include("recorders/LogZRecorder.jl")
export LogZRecorder
include("recorders/LossRecorder.jl")
export SKLRecorder, get_objective_vals, get_Λ_opt_round

# Statistics
include("stats/barriers.jl")
include("stats/stepping_stone.jl")
include("stats/record_lps.jl")
export get_round_lps
include("stats/stats.jl")
export round_trip_rate, count_chain_round_trips, count_round_trips_per_round
export round_trip_completion_iters

# Main algorithm
include("DEO.jl")
include("NRPTConfig.jl")
export NRPTConfig
include("nrpt.jl")
export nrpt, optimized_nrpt

# Problem library
include("problem_library/mvnormal.jl")
export mvnormal_slice_sampler, normal_gbm
include("problem_library/normal_iid.jl")
export normal_iid
include("problem_library/gmm.jl")
export gmm_slice_sampler, gmm_gbm
include("problem_library/cauchy.jl")
export cauchy_slice_sampler
include("problem_library/unidentifiable_product.jl")
export unidentifiable_product_slice_sampler, unidentifiable_product_gbm
include("problem_library/ode/transfection.jl")
export transfection_ode_slice_sampler, load_transfection_data, transfection_ode_gbm
include("problem_library/ising.jl")
export IsingModel, IsingGibbs


end
