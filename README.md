
# KernelMatrices.jl

This software suite is a companion to the manuscript [Scalable Gaussian Process
Computations using Hierarchical Matrices](https://arxiv.org/abs/1808.03215). It
is for working with kernel matrices where individual elements can be computed
efficiently, so that one can write ```julia A[i,j] = F(x[i], y[j], v), ``` Where
`x` and `y` are any kind of structure that allows getindex-style access (like a
vector) and `v` is a vector of parameters.  I have overloaded most (if not all)
of the relevant Base operations for the KernelMatrix struct, so that once you
create the lightweight KernelMatrix struct, call it `K`, with ```julia julia> K
= KernelMatrices.KernelMatrix(xpts, ypts, kernel_parameters, kernel_function)
``` you can access the (i,j)-th element of that matrix with `K[i,j]`, the j-th
column or row with `K[:,j]` and `K[j,:]`, get the full matrix with
`KernelMatrices.full(K)`. 

To accelerate linear algebra, this software package implements the
Hierarchically Off-Diagonal Low-Rank (HODLR) matrix structure originally
introduced in Ambikasaran and Darve (2013), which can be used to achieve
quasilinear complexity in matvecs, linear solves, and log-determinants.

For a brief whirlwind tour, here is a heavily commented example that builds
these objects. In the `./examples/` directory, you will find similar files that
you can run and mess with.
```julia
using LinearAlgebra, KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors

# Choose the number of locations:
n    = 1024

# For the example, randomly generate some locations in the form of a Vector{SVector{2, Float64}}.
# You don't need to use an SVector here---there are no restrictions beyond the locations being a
# subtype of AbstractVector. I just use StaticArrays for a little performance boost.
locs = [SVector{2, Float64}(randn(2)) for _ in 1:n]

# Declare a kernel function with this specific signature. If you want to use the HODLR matrix
# format, this function needs to be positive definite. It absolutely does NOT need a nugget-like
# term on the diagonal to achieve this, but if the positive definite kernelfunction is analytic
# everywhere, including at the origin, you may encounter some numerical problems. To avoid writing
# something like the Matern covariance function here, I pick a simpler positive definite function
# and simply add a nugget to be safe.
function kernelfunction(x::AbstractVector, y::AbstractVector, p::AbstractVector{T})::T where{T<:Number}
  out = abs2(p[1])/abs2(1.0 + abs2(norm(x-y)/p[2]))
  if x == y
    out += 1.0
  end
  return out
end

# Choose some parameters for the kernel matrix. These will go in with the p argument in kernelfunction.
kprm = SVector{2, Float64}(ones(2))

# Create the kernel matrix! You can basically treat this like a regular array and do things like
# K[i,j], K[i,:], K[:,j], and so on. I also have implemented things like K*vec, but I encourage you
# not to use them, because they will be slow and kind of defeat the purpose.
K    = KernelMatrices.KernelMatrix(locs, locs, kprm, kernelfunction)
```
If you want to make a HODLR matrix out of K, you have two low-rank approximation
methods for the off-diagonal blocks: you can use the adaptive cross
approximation (ACA) of Bebendorf (2000) and Rjasanow (2002), or you can use the
Nystrom approximation of Williams and Seeger (2001). I will show you the syntax
for both here, although they are very similar.

```julia
# For the ACA, you need to choose:
  # A relative preocision for the off-diagonal block approximation (tol),
  # A fixed level (lvl),
  # An optional fixed maximum rank (rnk),
  # A parallel assembly option (pll).
tol  = 1.0e-12            # This flag works how you'd expect.
lvl  = HODLR.LogLevel(8)  # Sets the level at log2(n)-8. HODLR.FixedLevel(k) also exists and works how you'd expect.
rnk  = 0                  # If set to 0, no fixed max rank. Otherwise, this arg works as you'd expect.
pll  = false              # This flag determines whether assembly of the matrix is done in parallel.
HK_a = HODLR.KernelHODLR(K, tol, rnk, lvl, nystrom=false, plel=pll)

# For the Nystrom approximation, you need to choose:
  # An optional fixed level (lvl),
  # A fixed off-diagonal rank that is GLOBAL (rnk),
  # and a parallel assembly option.
# Note that "tol" is not actually used here. I could probably clean that code up some day.
n_rk = 2 # choose a valid fixed rank for off-diagonal blocks
HK_n = HODLR.KernelHODLR(K, tol, n_rk, lvl, nystrom=true, plel=pll)
```
You can now do `HK_n*vec` and `HK_a*vec` in quasilinear complexity, assuming in
the case of `HK_a` that the rank of the off-diagonal blocks is O(log n) or less.
If you want to do the solves and logdets in that complexity as well, you will
need to compute the symmetric factorization.

Assuming that the output HODLR matrix is positive definite, you can factorize
the matrix easily.  This function modifies the struct internally, so this is all
you need to do:
```julia
HODLR.symmetricfactorize!(HK_n, plel=pll)
```
Now you can compute your linear solves with `HK_n\vec` and logdets with
`logdet(HK_n)`.

As a reminder, if you want to do things in on-node parallel, start julia with
multiple processes with `julia -p $k` for k many processes. If you just start
the REPL or run something with `julia ...`, then you will NOT really benefit
from using the parallel flag.

Finally, this software packages provides more specialized functionally for
Gaussian process computing. See the paper for details on this, but an
approximated Gaussian log-likelihood with a stochastic gradient, Hessian, and
expected Fisher information matrix is also provided. All of these things can be
computed in quasilinear time, so that many optimization options are available at
good complexity.

For details on this, I will send readers to the `./examples/fitting/` directory,
which has many complete and heavily commented scripts demonstrating how to do
that.


# Usage
This package is unregistered, so please install with
```julia
Pkg> add git://bitbucket.org/cgeoga/KernelMatrices.jl.git #v0.6.0
```
The figures and results of the paper were generated with version `0.3`, so if
you want to run that exact code again please add that version instead. If you
want to re-compute results from the paper and then potentially re-create the
figures, please run the files from the directory they are located in. All of
them use relative paths for output storage.

The code is organized into a module `KernelMatrices` and a sub-module
`KernelMatrices.HODLR`, which you can access with
```julia
julia> using KernelMatrices, KernelMatrices.HODLR
```
**For both modules, not everything is exported to the namespace, so you will
need to call some functions with `KernelMatrices.foo()`, or `HODLR.foo()`, or
`KernelMatrices.HODLR.foo()` if you do not bring the module `HODLR` into the
namespace with `using KernelMatrices.HODLR`. Please see the example files for
useful templates.**

**Please see the example files, which give complete examples of performing
optimization with first and second derivatives**.


# Changes to expect in the next release

1. **Small performance improvements.** In general, the code tries to be both fast and
   pedagogical. At some point, I am going to think more exclusively about being fast. Maybe I will
   make a separate branch to preserve the pedagogical codebase.

2. **Small readability improvements.** Some of this code was written well over a year ago, and I
   think I've gotten better at writing ergonomic and readable code. Again, in the interest of being
   sure that I'm releasing code that corresponds exactly to the results shown in the paper, I have
   not gone in and improved little guts in the code base. But in the next release, I will do that.


# Changes to expect eventually (HELP WANTED)

1. **Better parallelization.** I don't really know very much about parallel computing, and while
   I've tried to write this code in a way that suits parallelization, it doesn't make up for the
   fact that I don't really understand how to tune it or prepare it for 1000 cores or what have you.
   So, I'm hoping to either learn more about that or collaborate with people who do have knowledge
   in this area to see if this codebase can be made properly scalable to truly big metal.

2. **Integration with LowRankApprox.jl.** There are many beautiful matrix compression methods that
   are useful when you have a fast matvec instead of fast access to individual entries. Ken Ho wrote
   an exceptional software library for randomized methods that could be interfaced with this code to
   make more hierarchical matrix assembly methods possible. 

3. **More H-matrix formats.** I have structured the code to be easy to contribute to. If somebody
   wanted to write a module for working with HSS matrices, for example, they could make a sub-module
   HSS just like I already made a HODLR sub-module. 


# Authors

* Chris Geoga (cgeoga $at anl $dot gov)
* Paul Beckman (pbeckman $at anl $dot gov)


# Citation

If you use this software suite, please cite [this paper](https://doi.org/10.1080/10618600.2019.1652616).


# License

This software is distributed under the GNU GPL v2 license only. We are not
interested in redistributing it under any other license.

