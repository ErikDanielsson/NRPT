include("titanic-eval-sigma-10.jl")
import Dates

runs = Dict()
run_all(;runs=runs)
show_results(runs; dir=joinpath(@__DIR__, "results-$(Dates.today())"))