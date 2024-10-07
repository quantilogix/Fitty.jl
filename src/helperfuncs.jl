# This file is part of the Fitty.jl package
#
# © Raibatak Das & Quantilogix LLC, 2024
# MIT License

# Helper functions

datasets = Dict("DNase" => "../data/DNase.csv")

function loadDataset(name::String)
    local src
    try
        src = datasets[name]
    catch e
        throw(KeyError(name, "Not a valid dataset name"))
    end
    CSV.read(src, DataFrame)
end
    

"""Test fits with different scalings 
"""
function testFit(residualfunc::Function, guess::Union{Vector, NamedTuple}; 
                 data = nothing, weights = 1, lb = -Inf, ub = Inf, 
                 maxiter = 2000, kwargs...)
    # Fit with input guess for all 3 scalings
    scalings = ("", "unif", "diag")
    fits = []
    conv = [false, false, false]
    for (jj, sc) in enumerate(scalings)
        local fitresults
        try # Try each scaling and check if it works
            fitresults = nlsqfit(residualfunc, guess, data = data, 
                                 weights = weights,
                                 lb = lb, ub = ub, 
                                 maxiter = maxiter; 
                                 kwargs...,
                                 scaling = sc, quiet = true)
        catch e # If not, print a msg
            println("Fit with scaling = '", sc, "' failed for initial guess ", guess)
            @debug e
            fits = push!(fits, nothing)
        else
            fits = push!(fits, fitresults)
            # Check if fit converged
            conv[jj] = (fitresults.convergence != 0)
        end
    end
    if any(conv)
        # Identify fit with lowest SSR
        fits = fits[conv]
        scalings = scalings[conv]
        ssrVals = [fit.SSR for fit in fits]
        idx = argmin(ssrVals)
        bestfit = fits[idx]
        print("Fit with scaling = '", scalings[idx], "'")
        println(" converged in ", bestfit.nsteps, " steps for intial guess ", guess)
        println("Parameter estimates:")
        print(bestfit, "\n")
        println("RSE = ", bestfit.resStdErr, "\n")
        return fits[idx]
    else
        println("Fits did not converge in $maxiter iterations starting with intial guess $guess")
        println("Try a different starting guess, increase maxiter or increase tolerances")
        println()
    end
end


