## References for `Fitty.jl` package

**Raibatak Das - 2024-JAN-21**

This document contains references on optimization, trust region algorithms, bootstrap and Bayesian bootstrap methods and Julia package development. Also included are links to reference datasets used to benchmark `Fitty.jl`

### General reference

1. ["Numerical Optimization"](NocedalWright-2006-NumericalOptimization.pdf) by Nocedal and Wright (2006) <a href="https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf" target="_blank">pdf</a>

### Levenberg-Marquardt algorithm

1. ["The Levenberg-Marquardt Algorightm: Implementation and Theory"](More-1977-Levenberg-Marquardt.pdf), Jorge Moré (1977). Describes a single step of the algorithm and the two approaches to solving for the proposed displacement -using normal equations, or as a (linear) least squares problem. <a href="https://typeset.io/pdf/the-levenberg-marquardt-algorithm-implementation-and-theory-u1ziue6l3q.pdf" target="_blank"> pdf </a>
2. ["Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization"](Transtrum-Sethna-LM-2012.pdf). Proposes delayed gratification and geodesic acceleration terms to improve convergence. <a href="https://arxiv.org/abs/1201.5885" target="_blank"> pdf </a>
3. ["The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems"](Gavin-2019-LevenbergMarquardt.pdf) by Henry P. Gavin @ Duke University. Contains description of the algorithm and convergence criteria with links to MATLAB code. <a href="https://people.duke.edu/~hpgavin/m-files/lm.pdf" target="_blank">pdf</a>
4. Ceres solver documentation <a href="http://ceres-solver.org/index.html" target="_blank">Link</a>

### Bounded optimization, trust region and interior reflective algorithms 

1. ["A reflective Newton method or minimizing a quadratic function subject to bounds on some of the variables"](ColemanLi-1992-ReflectiveNewton.pdf) by Coleman and Li (1992) <a href="https://ecommons.cornell.edu/bitstream/1813/5486/1/92-111.pdf" target="_blank">pdf</a>
2. 

### Bayesian bootstrap

1. ["The Bayesian Bootstrap"](Rubin-1981-BayesianBootstrap.pdf), Rubin (1981). Original paper describing Bayesian bootstrap algorithm <a href="https://projecteuclid.org/euclid.aos/1176345338" target="_blank">pdf</a>
2.  Blog posts by Rasmus Bååth:
    - <a href="https://www.sumsar.net/blog/2015/04/the-non-parametric-bootstrap-as-a-bayesian-model/" target="_blank">"The Non-parametric Bootstrap as a Bayesian Model"</a>
    - <a href="https://www.sumsar.net/blog/2015/07/easy-bayesian-bootstrap-in-r/" target="_blank">"Easy Bayesian Bootstrap in R"</a>
3. R package `bayesboot` for Bayesian bootstrap <a href="https://cran.r-project.org/web/packages/bayesboot/readme/README.html">README</a> 
4. Blog post by Matteo Courthoud <a href="https://matteocourthoud.github.io/post/bayes_boot/" target="_blank">The Bayesian Bootstrap</a>

### Data sets

1. NIST statistical reference datasets for nonlinear regression <a href="https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml" target="_blank">Link</a> <br> Also see the following pages for more details:  
    - Background infomation <a href="https://www.itl.nist.gov/div898/strd/nls/nls_info.shtml" target="_blank">Link</a> 
    - Definitions <a href="https://www.itl.nist.gov/div898/strd/nls/data/LINKS/c-misra1a.shtml" target="_blank">link</a>
    - "Selection and Certification of Reference Datasets for Testing the Numerical Accuracy of Statistical Software for Non-linear Regression", Presentation by Janet Rogers <a href="https://www.itl.nist.gov/div898/strd/general/related/jsm97jr/nls_title.html" target="_blank">Slides</a>
2. "Nonlinear regression analysis and its applications" by Bates and Watts (1988)
    - [Appendix 1](BatesWatts-1988-Appendix1-ExampleDataSets.pdf) Data sets used in examples <a href="https://onlinelibrary.wiley.com/doi/epdf/10.1002/9780470316757.app1" target="_blank">pdf</a>
    - [Appendix 4](BatesWatts-1988-Appendix4-ProblemDataSets.pdf) Data sets used in problems <a href="https://onlinelibrary.wiley.com/doi/epdf/10.1002/9780470316757.app4" target="_blank">pdf</a>
3. ["Testing Unconstrained Optimization Software"](More-1981-TestingUnconstrained.pdf), Moré et al (1981) Collection of test functions to test optimization software <a href="https://dl.acm.org/doi/pdf/10.1145/355934.355936" target="_blank">Link</a>
4. Virtual library of simulation experiments - Optimization test problems <a href="https://www.sfu.ca/~ssurjano/optimization.html" target="_blank">Link</a>

### Other Julia packages for nonlinear regression

1. <a href="https://julianlsolvers.github.io/LsqFit.jl/latest/" target="_blank">LsqFit.jl</a>
2. Comparison of nonlinear solvers in Julia <a href="https://juliapackagecomparisons.github.io/pages/nonlinear_solvers/#nonlinear_least_squares_solvers" target="_blank">Link</a>
