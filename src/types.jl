# This file is part of the Fitty.jl package
#
# © Raibatak Das & Quantilogix LLC, 2024
# MIT License

# Define custom structs to store fit results
# and i/o functions to display results tables

""" Fit results with bootstrap CI estimates
"""
struct Fit
    parNames::Vector{String}
         fit::Union{NamedTuple, Vector{Float64}}
         SSR::Float64
       guess::Union{NamedTuple, Vector{Float64}}
          lb::Vector{Float64}
          ub::Vector{Float64}
    estimate::Vector{Float64}
      stdErr::Vector{Float64}
    normalCI::Matrix{Float64}
   conflevel::Float64
 convergence::Int
     #status::String
      nsteps::Int
        traj::Matrix{Float64}
     ssrvals::Vector{Float64}
   residuals::Vector{Float64}
        nobs::Int
    nparsFit::Int
         dof::Int
    redChiSq::Float64
         AIC::Float64
         BIC::Float64
   resStdErr::Float64
  covariance::Matrix{Float64}
       nBoot::Int
     BayesCI::Matrix{Float64}
   posterior::Matrix{Float64}
   bootChiSq::Vector{Float64}
 BCquantiles::Matrix{Float64}
      BCbias::Vector{Float64}
end

"""Fit results without bootstrap resampling
"""
struct BasicFit
    parNames::Vector{String}
         fit::Union{NamedTuple, Vector{Float64}}
         SSR::Float64
       guess::Union{NamedTuple, Vector{Float64}}
          lb::Vector{Float64}
          ub::Vector{Float64}
    estimate::Vector{Float64}
      stdErr::Vector{Float64}
    normalCI::Matrix{Float64}
   conflevel::Float64
 convergence::Int
      #status::String
      nsteps::Int
        traj::Matrix{Float64}
     ssrvals::Vector{Float64}
   residuals::Vector
        nobs::Int
    nparsFit::Int
         dof::Int
    redChiSq::Float64
         AIC::Float64
         BIC::Float64
   resStdErr::Float64
  covariance::Matrix{Float64}
end

"""Results of fit that failed to converge
"""
struct FailedFit
    parNames::Vector{String}
         fit::Union{NamedTuple, Vector{Float64}}
         SSR::Float64
       guess::Union{NamedTuple, Vector{Float64}}
          lb::Vector{Float64}
          ub::Vector{Float64}
    estimate::Vector{Float64}
 convergence::Int
      #status::String
      nsteps::Int
        traj::Matrix{Float64}
     ssrvals::Vector{Float64}
end

"""Construct table of fit results
"""
function resultsTable(r::Union{Fit, BasicFit})
    fitVals = r.estimate
    CI = (r isa Fit) ? r.BayesCI : r.normalCI
    g = r.guess
    guessVals = (g isa Vector) ? g : collect(values(g))
    npars = length(fitVals)
    if r.dof == 0 # Don't display ∞ standard errors and CIs
      return Table(Parameter = r.parNames,
                      Bounds = [[r.lb[j], r.ub[j]] for j = 1:npars],
                    Estimate = fitVals)
    end
    Table(Parameter = r.parNames,
             Bounds = [[r.lb[j], r.ub[j]] for j = 1:npars],
           Estimate = fitVals, 
             StdErr = r.stdErr,
                 CI = [CI[j, :] for j = 1:npars])
end

"Display detailed fit results"
function Base.show(io::IO, ::MIME"text/plain", r::Union{Fit, BasicFit})
    print("Fit results: ")
    display(resultsTable(r))
    n, mvar, dof = r.nobs, r.nparsFit, r.dof
    print("=========================================\n")
    println("Sum of squared residuals = $(@sprintf("%.3e", r.SSR))")
    println("Degrees of freedom = $n - $mvar = $dof")
    println("Residual standard error = $(@sprintf("%.3e", r.resStdErr))")
    if r isa Fit && r.dof != 0
        println("$(r.conflevel * 100)% Bayesian credible intervals computed using $(r.nBoot) bootstrap replicates")
    else
      if r.dof != 0
        println("$(r.conflevel * 100)% confidence intervals computed using normal approximation")
      end
    end
    println("=========================================")
end

"Display fitted parameter estimates"
Base.show(io::IO, r::Union{Fit, BasicFit}) = print(io, round.(r.estimate, digits = 3))

"""
Create DataFrame of posterior density  
"""
posteriorPDF(r::Fit) = DataFrame(r.posterior, r.parNames)
posteriorPDF(r::BasicFit) = error("Posterior distribution only available for fits called with bootstrap = true")
posteriorPDF(r::FailedFit) = error("Posterior distribution unavilable as fit did not converge")

