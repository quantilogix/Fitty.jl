# Julia module for nonlinear least squares fitting with
# Levenberg Marquardt algorithm using forward mode 
# automatic differentiation to compute gradients
#
# © Raibatak Das, 2023

module Fitty
using ForwardDiff, LinearAlgebra, Printf, REPL, TypedTables
import Distributions: cdf, Normal, quantile
import Statistics: mean

ltx = REPL.REPLCompletions.latex_symbols

function Fit(residualfunc::Function, guess::Vector, data; 
             bootstrap::Bool = false, nboot::Int64 = 2000, 
             conflevel = 0.95, kwargs...)
    # Define residual function for input data
    res = let f = residualfunc, df = data
        θ -> f(θ, df)
    end
    # Fit with input guess
    guess = convert(Vector{Float64}, guess)
    fit = LMfit(res, guess; kwargs...) 
    # Call bootstrap functions if requested provided fit converges
    if bootstrap
        if fit.convergence == 0
            println("Not running bootstrap as fit did not converge")
            return fit
        end
        bootPars = RunBoot(residualfunc, guess, data, nboot; kwargs...)
        jackPars = RunJackknife(residualfunc, guess, data; kwargs...)
        bootCI = ComputeBootCIs(fit.fit, bootPars, jackPars, conflevel)
        fit = (fit..., bootCI = bootCI, bootPars = bootPars)
        #return (fit, bootPars)
    end
    return fit
end

function TestFit(residualfunc::Function, guess::Vector, data; maxiter = 2000, kwargs...)
    # Define residual function for input data
    res = let f = residualfunc, df = data
        θ -> f(θ, df)
    end
    # Fit with input guess for all 3 scalings
    guess = convert(Vector{Float64}, guess)
    scalings = ("", "unif", "diag")
    fits = []
    conv = [false, false, false]
    for (jj, sc) in enumerate(scalings)
        local fit
        try # Try each scaling and check if it works
            fit = LMfit(res, guess; maxiter = maxiter,
                        kwargs..., scaling = sc, quiet = true)
        catch # If not, print a msg
            println("Fit with scaling = '", sc, "' failed for initial guess ", guess)
            fits = push!(fits, nothing)
        else
            fits = push!(fits, fit)
            # Check if fit converged
            conv[jj] = (fit.convergence != 0)
        end
    end
    if any(conv)
        # Identify fit with lowest SSR
        fits = fits[conv]
        scalings = scalings[conv]
        ssrVals = [fit.SSR for fit in fits]
        idx = argmin(ssrVals)
        bestFit = fits[idx]
        print("Fit with scaling = '", scalings[idx], "'")
        println(" converged in ", bestFit.nsteps, " steps for intial guess ", guess)
        println("Sum of squared residuals = $(@sprintf("%.3e", bestFit.SSR))")
        println("Residual standard error = $(@sprintf("%.3e", bestFit.resStdErr))")
        println("Parameters: ")
        m = size(guess, 1)
        parnames = ["θ$(ltx["\\_$j"])" for j=1:m]
        display(Table(Parameter = parnames, Guess = guess, 
                      Estimate = bestFit.fit, StdErr = bestFit.parStdErr))
        println()
        return fits[idx]
    else
        println("Fits did not converge in $maxiter iterations starting with intial guess $guess")
        println("Try a different starting guess, increase maxiter or increase tolerances")
        println()
    end
end


function OneLMstep(residualfunc::Function, guess::Vector{Float64}, 
                   λ::Float64, scaling::String = "unif")
    """Single Levenberg Marquardt step for input
    residual function, parameter guess and damping
    """
    # Compute residuals at input guess
    err = residualfunc(guess)
    # Compute Jacobian using ForwardDiff
    J = ForwardDiff.jacobian(residualfunc, guess)
    # Propose LM step
    d = sum(J.^2, dims = 1)[:]
    if scaling == "unif"
        μ = λ * maximum(d)
        LHS = [J; UniformScaling(√μ)]
    elseif scaling == "diag"
        LHS = [J; √λ * Diagonal(d)]
    else
        LHS = [J; UniformScaling(√λ)]
    end
    m = length(guess)
    RHS = [-err; zeros(m)]    
    return LHS \ RHS, J
end

SumOfSqrdRes(err::Vector{Float64}) = sum(err.^2)

"""Levenberg-Marquardt algorithm for nonlinear 
least squares regression"""
function LMfit(residualfunc::Function, guess::Vector{Float64};
               maxiter::Int64 = 2000,
               λ0             = 1e-3, 
               scaling        = "unif",
               update         = (decrease = 3, increase = 2),
               tolerance      = (func = 1e-12, pars = 1e-8, grad = 1e-8),
               quiet          = false)
    m = length(guess)
    traj = zeros(maxiter+1, m)
    traj[1,:] = guess
    # Compute sum of squared residuals at initial guess
    ssr = zeros(maxiter+1)
    err = residualfunc(guess)
    ssr[1] = SumOfSqrdRes(err)
    n = length(err)
    # Iteratively update guess
    tries, k, λ, convergence = 0, 1, λ0, 0
    while (convergence == 0) && (tries <= maxiter)
        # Propose move
        r = traj[k,:] 
        δ, J = OneLMstep(residualfunc, r, λ, scaling)
        p = r + δ
        # Compute SSR after proposed step
        err = residualfunc(p)
        newssr = SumOfSqrdRes(err)
        # Compute ratio of observed to predicted ΔSSR
        ΔSSRobserved = ssr[k] - newssr
        ΔSSRpredicted = sum((J*δ).^2)
        ρ = ΔSSRobserved/ΔSSRpredicted
        if (newssr <= tolerance.func) || (ρ > 0.1) # Accept 
            k += 1
            traj[k, :] = p 
            ssr[k] = newssr
            # Decrease damping if good move
            if ρ > 0.75
                λ /= update.decrease
            end
            # Check convergence in SSR
            if  (newssr <= tolerance.func) || (ΔSSRobserved <= tolerance.func * (1 + ssr[k-1]))
                convergence = 1
            end
            # Check convergence in parameter estimates
            δmax = maximum(abs.(δ))
            if δmax <= tolerance.pars * (1 + maximum(abs.(r)))
                convergence = 2
            end
            # Check convergence in gradient
            grad = maximum(abs.(J*r))
            if grad <= tolerance.grad
                convergence = 3
            end
        else # Reject move and increase damping
            λ *= update.increase
        end
        tries += 1
    end
    if convergence == 0 # Fit did not converge
        if !(quiet)
            println("Fit did not converge in $maxiter iterations starting with intial guess $guess")
            println("Try a different starting guess, increase maxiter, or increase tolerances")
            println()
        end
        return(        fit = traj[k, :], 
                       SSR = ssr[k], 
               convergence = convergence, 
                      traj = traj[1:k,:], 
                    values = ssr[1:k]);
    else # Success!
        # Compute residuals at fitted value
        fit = traj[k, :]
        err = residualfunc(fit)
        dof = length(err) - m
        chisq = ssr[k]/dof
        # Compute covariance matrix and standard errors
        J = ForwardDiff.jacobian(residualfunc, fit)
        cov = chisq*inv(J'*J)
        stderr = sqrt.(diag(cov))
        # Print parameter table to console
        if !(quiet)
            println("Fit converged in $k steps from intial guess ", guess)
            println("Sum of squared residuals = $(@sprintf("%.3e", ssr[k]))")
            println("Residual standard error = $(@sprintf("%.3e", chisq))")
            println("Parameters: ")
            parnames = ["θ$(ltx["\\_$j"])" for j=1:m]
            display(Table(Parameter = parnames, Guess = guess, Estimate = fit, StdErr = stderr))
            println()
        end
        return (        fit = fit,
                  parStdErr = stderr,
                        SSR = ssr[k], 
                convergence = convergence,
                     nsteps = k,
                       traj = traj[1:k,:], 
                     values = ssr[1:k],
                  residuals = err,
                        dof = dof,
                   redChiSq = chisq,
                  resStdErr = sqrt(chisq),
                        cov = cov);
    end
end # End of function LMfit

function RunBoot(residualfunc::Function, guess::Vector{Float64}, data, 
                 nboot::Int64; kwargs...)
    npars = size(guess, 1)
    bootPars = zeros(Float64, nboot, npars)
    maxtries = 2*nboot
    # Generate bootstrap fits with resampled data
    println("Generating ", nboot, " bootstrap fits... ")
    k, tries = 0, 0
    while (k < nboot) && (tries < maxtries)
        rdf = Resample(data)
        local fit
        try
            fit = LMfit(θ -> residualfunc(θ, rdf), guess; kwargs..., quiet = true)
        catch e
        else
            if fit.convergence != 0
                k += 1;
                bootPars[k, :] = fit.fit
            end
        end
        tries += 1
    end
    println(k, " bootstrap fits converged out of ", tries, " tries \n")
    return bootPars[1:k, :]
end # End of function RunBoot
    
# Resample data with replacement
function Resample(df)
    n = size(df, 1)
    idx = sort([rand(1:n) for j = 1:n])
    return df[idx, :]
end

function RunJackknife(residualfunc::Function, guess::Vector{Float64}, data; kwargs...)
    # Compute Jackknife estimates by fitting all but one datapoint
    n, m = size(data, 1), size(guess, 1)
    jack = zeros(Float64, n, m)
    conv = falses(n)
    for j = 1:n
        df = data[1:n .!= j, :] # All but j-th row
        local fit
        try
            fit = LMfit(θ -> residualfunc(θ, df), guess; kwargs..., quiet = true)
        catch e
        else
            if fit.convergence != 0
                conv[j] = true
                jack[j, :] = fit.fit
            end
        end
    end
    return jack[conv, :]
end

function ComputeBootCIs(fitPars::Vector{Float64}, bootPars::Array{Float64, 2}, jackPars::Array{Float64, 2}, conflevel::Float64)
    # Compute bootstrap confidence intervals for input
    # confildence level
    if !(0 < conflevel < 1)
        error("Confidence level must lie between 0 and 1")
    end
    npars = length(fitPars)
    nboot = size(bootPars, 1)
    # Compute percentile confidence intervals
    alpha = (1 - conflevel)/2
    qtiles = [alpha, 1 - alpha]
    percentiles = zeros(Float64, npars, 2)
    for k = 1:npars
        percentiles[k,:] = quantile(bootPars[:, k], qtiles)
    end
    # Compute BCa confidence intervals
    BCa = zeros(Float64, npars, 2)
    for k = 1:npars
        f, b, j = fitPars[k], bootPars[:, k], jackPars[:, k] 
        # Bias correction
        z0 = quantile(Normal(), sum(b .< f)/nboot)
        # Acceleration
        m = mean(j)
        a = sum((j .- m).^3) / (6 * (sum((j .- m).^2))^(3/2))
        zAlpha = quantile(Normal(), qtiles)
        zLo = z0 + (z0 + zAlpha[1]) / (1 - a*(z0 + zAlpha[1]))
        zHi = z0 + (z0 + zAlpha[2]) / (1 - a*(z0 + zAlpha[2]))
        BCaqtiles = cdf(Normal(), [zLo, zHi])
        BCa[k, :] = quantile(bootPars[:, k], BCaqtiles)
    end
    return (BCa = BCa, Percentile = percentiles)
end

end # End of module