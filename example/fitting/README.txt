
These are a handful of example scripts that are specific to the application of
fitting Gaussian process data. A brief list:

fitting_funs.jl: a simple collection of covariance functions and their first and
second derivatives, which are necessary for the gradient/expected fisher and
Hessian (respectively).

generic_exact_functions.jl: The exact versions of the negative Gaussian
log-likelihood, its gradient, and its Hessian, as well as the exact HODLR
versions of those (as in, ones that don't use stochastic trace estimation).
These are not particularly tuned for performance, and if you were to use these
as a convenient base for writing your own exact functions, I would strongly
suggest thinking about being a bit more thoughtful about repeated computations
than I was.

test_fit_*.jl: A few examples that have the small boilerplate required to fit
data in a few ways. The nlopt example uses the method of moving asymptotes,
which does not use the Hessian; The trustregion example minimizes the full
likelihood AND profile likelihood with the trust region method and compares the
two results.


The boilerplate provided by the test_fit_*.jl files should really be enough to
cover basically any use case of the HODLR code in this repository. The files in
../paperscripts/ basically copy this boilerplate in a less verbose way, so if
anything there isn't clear, I would suggest coming back to these files.

