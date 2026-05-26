include("titanic-eval-sigma-hyperprior.jl")
import Dates

runs = Dict()
run_all(;runs=runs)
show_results(runs; joinpath(@__DIR__, "results-$(Dates.today())"))