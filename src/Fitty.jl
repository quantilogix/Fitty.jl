# Julia module for nonlinear least squares fitting
# with parameter bounds using a trust region 
# interior reflective algorithm, forward mode 
# automatic differentiation  to compute gradients,
# and a Bayesian bootstrap procedure to simulate 
# parameter posterior distributions and compute 
# confidence/credible intervals (CI)
#
# © Raibatak Das & Quantilogix LLC, 2024
# MIT License


module Fitty

using CSV, DataFrames, ForwardDiff, FiniteDifferences, LinearAlgebra, Printf, TypedTables
using Distributions: cdf, Dirichlet, Normal, quantile, TDist
using Statistics: mean
using StatsBase: ecdf
import NaNMath as nm

include("types.jl")
include("residualfuncs.jl")
include("fittingfuncs.jl")
include("bootstrapfuncs.jl")
include("helperfuncs.jl")

end # End of module
