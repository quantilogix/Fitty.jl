# This file is part of the Fitty.jl package
#
# © Raibatak Das & Quantilogix LLC, 2024
# MIT License

# Define functions to run Bayesian bootstrap and compute 
# parameter CIs

"""Simulate Bayesian bootstrap procedure to produce parameter posterior
distribution by repeated weighted fits
"""
function runBayesBoot(residualfunc::Function, guess::Vector{Float64}, 
                      lb::Vector{Float64}, ub::Vector{Float64}, 
                      nboot::Int64; DirichletAlpha = 4, kwargs...)
    npars = size(guess, 1)
    bootPars = zeros(nboot, npars)
    chisqVals =  zeros(nboot)
    maxtries = 2*nboot
    # Define function to compute unweighted residuals
    fSSR = sumSqrdRes ∘ residualfunc
    # Set up Dirichlet distribution to compute weights
    nobs = length(residualfunc(guess))
    weightDist = Dirichlet(nobs, DirichletAlpha)
    # Define weighted residual function for bootstrap replicates
    res(θ, weights) = sqrt.(weights) .* residualfunc(θ)
    # Generate bootstrap fits with weighted residuals
    @debug "Generating $nboot Bayesian bootstrap fits..."
    k, tries = 0, 0
    while (k < nboot) && (tries < maxtries)
        # Draw weights
        w = rand(weightDist)
        local fit
        try
          fit = tirFit(θ -> res(θ, w), guess, lb, ub; kwargs..., quiet = true)
        catch e
        else
            if fit.convergence > 0
                k += 1;
                bootPars[k, :] = fit.fit
                chisqVals[k] = fSSR(fit.fit) / fit.dof
            end
        end
        tries += 1
    end
    println(k, " bootstrap fits converged out of ", tries, " tries")
    if tries >= maxtries
        println("Maximum number (", tries, ") of bootstrap fits attempted")
        println("but fewer than ", nboot, " converged.")
    end
    return bootPars[1:k, :], chisqVals[1:k]
end # End of function RunBayesBoot
    
"""Compute bias-corrected Bayesian credible intervals from Bayesian bootstrap sample
"""
function bayesBootCI(fitPars::Vector{Float64}, bootPars::Array{Float64, 2}, conflevel::Float64)
    # Check confidence level 
    if !(0 < conflevel < 1)
        error("Confidence level = $conflevel does not lie between 0 and 1")
    end
    npars = length(fitPars)
    nboot = size(bootPars, 1)
    # Compute percentiles for given confidence level
    alpha = (1 - conflevel)
    qtiles = [alpha/2, 1 - alpha/2]
    # Compute bias corrected credible interval
    local bias = zeros(Float64, npars)
    local BCquantiles = zeros(Float64, npars, 2)
    local BayesCI = zeros(Float64, npars, 2)
    if !isempty(bootPars)
      for k = 1:npars
          f, b = fitPars[k], bootPars[:, k]
          # What fraction of bootstrap  estimates 
          # are <= estimate in original fit
          z0 = quantile(Normal(), sum(b .<= f)/nboot)
          bias[k] = z0
          # Apply bias correction
          zBC = 2*z0 .+ quantile(Normal(), qtiles)
          BCquantiles[k, :] = cdf(Normal(), zBC)
          # Compute bias corrected credible interval
          BayesCI[k, :] = quantile(bootPars[:, k], BCquantiles[k,:])
      end
    end
    return (BayesCI = BayesCI, BCquantiles = BCquantiles, bias = bias)
end

