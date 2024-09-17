# This file is part of the Fitty.jl package
#
# © Raibatak Das & Quantilogix LLC, 2024
# MIT License

# Define fitting functions

"""
Ordinary or weighted nonlinear least squares fit 
with support for parameter bounds, and Bayesian bootstrap
resampling to compute parameter posterior distributions 
and credible intervals

    nlsfit(residualfunc::Function, guess::Union{Vector, NamedTuple}; 
           data = nothing, 
           weights = 1, 
           lb = -Inf, ub = Inf,
           bootstrap::Bool = false, 
           nboot::Int = 2000, 
           conflevel = 0.95, 
           quiet = false, 
           kwargs...)

Minimize the input residual function `residualfunc` starting
with the initial guess `guess`. The residual function takes
a parameter vector or named tuple as input and outputs a 
vector of residuals. `nlsfit` minimizes the sum of squared 
residuals and returns best fit parameter estimates

See https://quantilogix.io/Fitty for detailed usage examples

# Quickstart - Rosenbrock function
```julia-repl
julia> # Define residuals for Rosenbrock function
       f(θ) = [1 - θ.x, 10*(θ.y - θ.x^2)]
f (generic function with 1 method)

julia> # Minimize SSR starting with a non-optimal guess
       guess = (x = -1.5, y = 1); # Starting guess away from global minima

julia> fit = Fitty.nlsfit(f, guess);
Fit converged in 13 steps from intial guess [-1.5, 1.0]
to final estimate [1.0, 1.0]
```
"""
function nlsfit(residualfunc::Function, guess::Union{Vector, NamedTuple}; 
                 data = nothing, weights = 1, lb = -Inf, ub = Inf,
                 bootstrap::Bool = false, nboot::Int = 2000, 
                 conflevel = 0.95, quiet = false, kwargs...)
    # Check bounds for consistency
    npars = length(guess)
    # Construct bounds vectors
    boundsvector(x) = (x isa Number) ? x * ones(npars) : Float64.(x)
    lb, ub = boundsvector(lb), boundsvector(ub)
    @debug "Setting lower bound = $lb \nSetting upper bound = $ub"
    # Check bounds vector lengths
    if !(npars == length(lb) == length(ub))
        throw(DimensionMismatch("""Lengths of guess vector and bounds do not match. 
                # of parameters = $npars, 
                Length of lower bound = $(length(lb)), 
                Length of upper bound = $(length(ub))"""))
    end
    # Create vector of initial guesses
    guessVals = (guess isa Vector) ? Float64.(guess) : Float64.(collect(values(guess)))
    # Check that initial guess is within bounds
    if !all(lb .<= guessVals .<= ub)
        throw(ArgumentError("""Initial guess $(guess) lies outside the
                input lower bound  $(lb) and upper bound $(ub)}
                """))
    end
    # Construct residual function from inputs
    if weights isa Number # <-- Unweighted least squares
        if data == nothing
            res = (guess isa Vector) ? residualfunc : makeResidual(residualfunc, guess)
        else
            res = makeResidual(residualfunc, guess, data)
        end
    else # <-- Weighted least squares
        # Check for all positive weights
        if any(weights .< 0)
          throw(ArgumentError("Weights must be non-negative"))
        end
        # Check for correct length of weights vector
        residuals = (data == nothing) ? residualfunc(guess) : residualfunc(guess, data)
        nres = length(residuals)
        nwts = length(weights)
        if nres != nwts
            throw(DimensionMismatch("""Length of residual vector = $nres 
                    does not match length of weights vector = $nwts"""))
        else
            res = makeResidual(residualfunc, guess, weights, data = data)
        end
    end
    # Perform fit
    fit = tirFit(res, guessVals, lb, ub; quiet = quiet, kwargs...)
    # Assign parameter names to fit output
    if guess isa NamedTuple
        parSymbols = keys(guess)
        fitPars = NamedTuple(zip(parSymbols, fit.fit))
        parNames = collect(String.(parSymbols))
    else
        parNames = ["θ[$j]" for j in 1:npars]
        fitPars = fit.fit
    end
    # Gather results
    g = (guess isa Vector) ? guessVals : NamedTuple(zip(parSymbols, guessVals))
    if fit.convergence <= 0 # Fit failed to converge
        results = FailedFit(parNames, fitPars, fit.SSR, g, lb, ub,
                            fit.fit, fit.convergence, #fit.status, 
                            fit.nsteps, fit.traj, fit.ssrvals)
        return results
    else
        # Compute confidence intervals using normal approximation
        if fit.dof == 0
          local normalCI = fit.fit .+ [fit.stdErr -fit.stdErr]
        else
          prob = (1 - conflevel)/2
          z = quantile(TDist(fit.dof), prob)
          local normalCI = fit.fit .+ [z * fit.stdErr -z * fit.stdErr]
        end
        results = BasicFit(parNames, fitPars, 
                           fit.SSR, g, lb, ub,
                           fit.fit, fit.stdErr, normalCI, conflevel,
                           fit.convergence, #fit.status, 
                           fit.nsteps, fit.traj, fit.ssrvals,
                           fit.residuals, fit.nobs, fit.nparsFit, fit.dof, 
                           fit.redChiSq, fit.AIC, fit.BIC, fit.resStdErr, fit.cov)
    end
    # Call bootstrap function if requested
    if bootstrap
        bootPars, bootChiSq = runBayesBoot(res, guessVals, lb, ub, nboot; kwargs...)
        bootCI = bayesBootCI(fit.fit, bootPars, conflevel)
        results = Fit(parNames, fitPars, 
                      fit.SSR, g, lb, ub,
                      fit.fit, fit.stdErr, normalCI, conflevel,
                      fit.convergence, #fit.status, 
                      fit.nsteps, fit.traj, fit.ssrvals,
                      fit.residuals, fit.nobs, fit.nparsFit, fit.dof, 
                      fit.redChiSq, fit.AIC, fit.BIC, fit.resStdErr, fit.cov, 
                      nboot, bootCI.BayesCI, bootPars, bootChiSq, 
                      bootCI.BCquantiles, bootCI.bias)
    end
    if !(quiet)
        #println("to final estimate ", results)
        if data != nothing
            display(results)
        end
    end
  return results
end

"""Create Jacobian function for input residual function
"""
function makeJac(res::Function)
    function jac(x::Vector{Float64})
        try 
            ForwardDiff.jacobian(res, x)
        catch e
            @debug "ForwardDiff failed to compute Jacobian at $x. See error below:"
            @debug e
            @debug "Using FiniteDifferences to compute Jacobian"
            jacobian(central_fdm(5, 1), res, x)[1]
        end
    end
end

"""Compute a single step of the trust-region interior 
reflection algorithm for input residual function, 
current guess, damping parameter λ, and scaling
"""
function onestep(res::Function, jac::Function, guess::Vector{Float64}, 
                 idxFixed::Vector{Int64}, λ::Float64, 
                 scaling::String = "unif")
    # Compute residuals at input guess
    err = res(guess)
    # Compute Jacobian 
    #fullJ = ForwardDiff.jacobian(res, guess)
    fullJ = jac(guess)
    # Remove columns for fixed parameters
    if !isempty(idxFixed)
        J = fullJ[:, 1:end .∉ [idxFixed]]
    else
        J = fullJ
    end
    # Propose trust region step for reduced problem
    d = sum(J.^2, dims = 1)[:]
    if scaling == "unif"
        μ = λ * maximum(d)
        LHS = [J; UniformScaling(√μ)]
    elseif scaling == "diag"
        LHS = [J; √λ * Diagonal(d)]
    else
        LHS = [J; UniformScaling(√λ)]
    end
    m = length(guess) - length(idxFixed)
    RHS = [-err; zeros(m)]
    δ = LHS \ RHS # Proposed move 
    # Construct step with 0s along fixed parameter directions
    step = zeros(size(guess))
    step[1:end .∉ [idxFixed]] .= δ
    return step, fullJ
end

"""Compute sum of squared residuals
"""
sumSqrdRes(err::Vector{Float64}) = sum(err.^2)

"""Trust region-interior reflective algorithm for
least squares fitting with parameter bounds"""
function tirFit(residualfunc::Function, guess::Vector{Float64},
                lb::Vector{Float64}, ub::Vector{Float64};
                maxiter::Int64 = 2000,
                λ0             = 1e-3, 
                scaling        = "unif",
                update         = (decrease = 3, increase = 2),
                tolerance      = (func = 1e-12, step = 1e-8, grad = 1e-8),
                quiet          = false)
    m = length(guess)
    traj = zeros(maxiter+1, m)
    ssr = zeros(maxiter+1)
    if all(isinf.(lb)) && all(isinf.(ub))
        checkBounds = false
    else
        checkBounds = true
    end
    # Identify parameters held fixed vs varied
    idxFixed = findall(lb .== ub) # Indices of fixed parameters
    idxVar = findall(lb .!= ub) # Indices of variable parameters
    # Compute sum of squared residuals at initial guess
    fSSR = sumSqrdRes ∘ residualfunc
    traj[1,:] = guess
    ssr[1] = fSSR(guess)
    @debug "Initial guess: $guess with SSR = $(ssr[1])"
    # Define Jacobian function
    jacfunc = makeJac(residualfunc)
    # Iteratively update guess
    tries, k, λ, convergence = 1, 1, λ0, 0
    while (convergence == 0) && (tries <= maxiter)
        # Propose move
        r = traj[k,:] 
        δ0, J = onestep(residualfunc, jacfunc, r, idxFixed, λ, scaling)
        p0 = r + δ0
        @debug "Try #$tries: Proposed step: $(r[idxVar]) + $(δ0[idxVar]) -> $(p0[idxVar])" 
        # Check for bounds
        if checkBounds && !all(lb .<= p0 .<= ub)
            @debug "$p0 lies outside bounds"
            # Find reflected point
            p1 = constrain.(p0, lb, ub)
            ssr1 = fSSR(p1)
            @debug "Reflected point: $p1 with SSR = $ssr1"
            # Find point near closest boundary 
            # along proposed direction
            p2, fStep = findNearest(p0, r, lb, ub)
            p2[idxFixed] .= guess[idxFixed]
            if fStep != 0
                ssr2 = fSSR(p2)
                @debug "Nearest point within bounds: $p2 with SSR = $ssr2"
                # Choose point with lower SSR
                p = (ssr1 < ssr2) ? p1 : p2
            else
                @debug "Current position at or near a bound. Using reflected point as proposal"
                p = p1
            end
            δ = p - r
        else
            p, δ = p0, δ0
        end
        # Compute SSR after proposed step
        newssr = fSSR(p)
        # Compute ratio of observed to predicted ΔSSR
        ΔSSRobserved = ssr[k] - newssr
        ΔSSRpredicted = sum((J*δ).^2)
        ρ = ΔSSRobserved/ΔSSRpredicted
        @debug("""SSR at current position $r =  $(ssr[k])
            SSR at proposed position $p = $newssr
            ΔSSRobserved = $ΔSSRobserved \t ΔSSRpredicted = $ΔSSRpredicted
            ρ = ΔSSRobserved/ΔSSRpredicted = $ρ """)
        if (newssr < tolerance.func) || (ρ > 0.1) # Accept
            @debug("Move accepted because ρ > 0.1 or new SSR < $(tolerance.func)")
            k += 1
            traj[k, :] = p 
            ssr[k] = newssr
            # Increase trust region radius if good move
            if (ρ > 0.75) 
                λ /= update.decrease 
                @debug("ρ > 0.75. Increasing trust region radius")
            end
            # Check convergence in SSR
            if  (newssr < tolerance.func) || (ΔSSRobserved < tolerance.func * (1 + ssr[k-1]))
                convergence = 1
            end
            # Check convergence in parameter estimates
            δmax = maximum(abs.(δ))
            if δmax < tolerance.step * (1 + maximum(abs.(r)))
                convergence = 2
            end
            # Check convergence in gradient
            grad = maximum(abs.(J*r))
            if grad < tolerance.grad
                convergence = 3
            end
        else # Reject move and increase λ to decrease trust region
            @debug("Update rejected because ρ < 0.1\t Decreasing trust region radius")
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
                    nsteps = k,
                      traj = traj[1:k,:], 
                   ssrvals = ssr[1:k]);
    else # Success!
        # Compute residuals at fitted value
        fit = traj[k, :]
        err = residualfunc(fit)
        n = length(err)
        # Compute degrees of freedom taking into account fixed parameters
        mvar = length(idxVar)
        dof = n - mvar
        # Compute reduced chi square, RSE, AIC and BIC
        χ2ν = ssr[k]/dof
        rse = sqrt(χ2ν)
        aic = 2*mvar + n * log(ssr[k])
        bic = mvar * log(n) + n * log(ssr[k])
        # Compute covariance matrix
        #J = ForwardDiff.jacobian(residualfunc, fit)
        J = jacfunc(fit)
        local cov
        if mvar == m
            cov = χ2ν * inv(J'*J)
        else
            cov = zeros(m, m)
            Jvar = J[:, idxVar]
            Cvar = χ2ν * inv(Jvar'*Jvar)
            cov[idxVar, idxVar] .= Cvar
        end
        # Compute standard error
        stderr = sqrt.(diag(cov))
        # Print parameter table to console
        if !(quiet)
            println("Fit converged in $(k-1) steps from intial guess ", guess)
            println("to final estimate ", round.(fit, digits = 3))
        end
        return (        fit = fit,
                        SSR = ssr[k], 
                     stdErr = stderr,
                convergence = convergence,
                     nsteps = k,
                       traj = traj[1:k,:], 
                    ssrvals = ssr[1:k],
                  residuals = err,
                       nobs = n,
                   nparsFit = mvar,
                        dof = dof,
                   redChiSq = χ2ν,
                        AIC = aic,
                        BIC = bic,
                  resStdErr = rse,
                        cov = cov);
    end
end

"""Constrain input value to lie between lower and 
upper bounds lb and ub by repeated reflection
"""
function constrain(x::Float64, lb::Float64, ub::Float64)
   if lb == ub
        @debug "lower bound = upper bound = $lb"
        return lb
    end
    if isnan(x)
        @debug "x = NaN ∴ constraint can not be satisfied"
        return x
    end
    if lb > ub
        @debug "Switching bounds from [$lb, $ub] to [$ub, $lb]"
        lb, ub = ub, lb
    end
    while !(lb <= x <= ub)
        if x > ub
            @debug "$x > upper bound = $ub"
            x = 2*ub - x
            @debug "Reflected to $x"
        else
            @debug "$x < lower bound = $lb"
            x = 2*lb - x
            @debug "Reflected to $x"
        end
    end
    return x
end

"""Find point within bounds close to the nearest 
boundary along the path r->x"""
function findNearest(x::Vector{Float64}, r::Vector{Float64}, 
                     lb::Vector{Float64}, ub::Vector{Float64})
    npars = length(x)
    f = zeros(npars)
    for j = 1:npars
        if x[j] < lb[j]
            f[j] = (r[j] - lb[j])/(r[j] - x[j])
        elseif x[j] > ub[j]
            f[j] = (ub[j] - r[j])/(x[j] - r[j])
        else
            f[j] = 1 # Full step
        end
    end
    # Identify nearest boundary
    fmin, j = findmin(abs.(f))
    @debug "Parameter value $(x[j]) is out of bounds"
    # Walk back step to inside nearest boundary
    fmin = max(0, prevfloat(fmin))
    @debug "Fractional step size to stay within bounds = $fmin"
    return @. r + fmin*(x - r), fmin
end

