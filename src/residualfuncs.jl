# This file is part of the Fitty.jl package
#
# © Raibatak Das, Quantilogix LLC, 2024
# MIT License

# Define functions to create residual function
# to pass to tirFit() from input to nlsqfit()

"""
Construct residual function from inputs
"""
function makeResidual(residualfunc::Function, guess::NamedTuple)
    # Construct residual function that accepts 
    # vector input
    res = let f = residualfunc, p = keys(guess)
        θ -> f(NamedTuple(zip(p, θ)))
    end
    return res
end

"""
Construct residual function from inputs
"""
function makeResidual(residualfunc::Function, guess::Vector, data)
    res = let f = residualfunc, d = data
        θ -> f(θ, d)
    end
    return res
end

function makeResidual(residualfunc::Function, guess::NamedTuple, data)
    res = let f = residualfunc, p = keys(guess), d = data
        θ -> f(NamedTuple(zip(p, θ)), d)
    end
    return res
end

"""
Construct residual function from inputs
"""
function makeResidual(residualfunc::Function, guess::Vector, 
                      weights::Vector; data = nothing)
    if data == nothing
      res = let f = residualfunc, w = sqrt.(weights)
        θ -> w .* f(θ)
      end
    else
        res = let f = residualfunc, d = data, w = sqrt.(weights)
          θ -> w .* f(θ, d)
        end
    end
    return res
end

"""
Construct residual function from inputs
"""
function makeResidual(residualfunc::Function, guess::NamedTuple, 
                      weights::Vector; data = nothing)
    if data == nothing
      res = let f = residualfunc, p = keys(guess), w = sqrt.(weights)
          θ -> w .* f(NamedTuple(zip(p, θ)))
        end
    else
      res = let f = residualfunc, p = keys(guess), d = data, w = sqrt.(weights)
          θ -> w .* f(NamedTuple(zip(p, θ)), d)
        end
    end
    return res
end

