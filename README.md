
# KernelMatrices.jl

This software suite is a companion to [this paper](https://arxiv.org/abs/1808.03215). It is for
working with kernel matrices where individual elements can be computed efficiently, so that one can
write
```julia
A[i,j] = F(x[i], y[j], v),
```
Where `x` and `y` are any kind of structure that allows getindex-style access (like a vector) and
`v` is a vector of parameters.  I have overloaded most (if not all) of the relevant Base operations
for the KernelMatrix struct, so that once you create the lightweight KernelMatrix struct, call it
`K`, with 
```julia
julia> K = KernelMatrices.KernelMatrix(xpts, ypts, kernel_parameters, kernel_function)
```
you can access the (i,j)-th element of that matrix with `K[i,j]`, the j-th column or row with
`K[:,j]` and `K[j,:]`, get the full matrix with `full(K)`. 

To accelerate linear algebra, this software package implements the Hierarchically Off-Diagonal
Low-Rank (HODLR) matrix structure originally introduced in Ambikasaran and Darve (2013), which can
be used to achieve quasilinear complexity in matvecs, linear solves, and log-determinants.

For a brief whirlwind tour, here is a heavily commented example that builds these objects. In the
`./examples/` directory, you will find similar files that you can run and mess with.
```julia
using KernelMatrices, KernelMatrices.HODLR, StaticArrays, NearestNeighbors

# Choose the number of locations:
n    = 1024

# For the example, randomly generate some locations in the form of a Vector{SVector{2, Float64}}.
# You don't need to use an SVector here---there are no restrictions beyond the locations being a
# subtype of AbstractVector. I just use StaticArrays for a little performance boost.
locs = map(x->SVector{2, Float64}(randn(2)), 1:n)

# Declare a kernel function with this specific signature to enforce type stability. If you want to
# use the HODLR matrix format, this function needs to be positive definite. It absolutely does NOT
# need a nugget-like term on the diagonal to achieve this, but if the positive definite
# kernelfunction is analytic everywhere, including at the origin, you may encounter some numerical
# problems. To avoid writing something like the Matern covariance function here, I pick a simpler
# positive definite function and simply add a nugget to be safe.
function kernelfunction{T<:Number}(x::Abstractvector, y::AbstractVector, p::AbstractVector{T})::T
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

If you want to make a HODLR matrix out of K, you have two low-rank approximation methods for the
off-diagonal blocks: you can use the adaptive cross approximation (ACA) of Bebendorf (2000) and
Rjasanow (2002), or you can use the Nystrom approximation of Williams and Seeger (2001). I will show
you the syntax for both here, although they are very similar.

```julia
# For the ACA, you need to choose:
  # A relative preocision for the off-diagonal block approximation (tol),
  # An optional fixed level (lvl),
  # An optional fixed maximum rank (rnk),
  # A parallel assembly option (pll).
tol  = 1.0e-12  # This flag works how you'd expect.
lvl  = 0        # an lvl value of 0 sets the level as log2(n)-8. You need the level to grows as
                # O(log(n)) for quasilinear complexity. See the paper for details.
rnk  = 0        # If set to 0, no fixed max rank. Otherwise, this arg works as you'd expect.
pll  = false    # This flag also works how you'd expect.
HK_a = HODLR.KernelHODLR(K, tol, lvl, rnk, nystrom=false, plel=pll)

# For the Nystrom approximation, you need to choose:
  # An optional fixed level (lvl),
  # A fixed off-diagonal rank that is GLOBAL (rnk),
  # and a parallel assembly option.
HK_n = HODLR.KernelHODLR(K, tol, lvl, rnk, nystrom=true, plel=pll)

# Congrats! You can now do HK_n*vec and HK_a*vec in quasilinear complexity. If you want to do the
# solves and logdets in that complexity as well, you'll need to commpute the symmetrix factorization.

# Assuming that the output HODLR matrix is positive definite, you can factorize the matrix easily.
# This function modifies the struct internally, so this is all you need to do.
HODLR.symmetricfactorize!(HK_n, plel=pll)

# Now, you can compute your linear solves with HK_n\vec and logdets with logdet(HK_n).
```

Finally, this software packages provides more specialized functionally for Gaussian process
computing. See the paper for details on this, but an approximated Gaussian log-likelihood with a
stochastic gradient, Hessian, and expected Fisher information matrix is also provided. All of these
things can be computed in quasilinear time, so that many optimization options are available at good
complexity.

For details on this, I will send readers to the `./examples/fitting/` directory, which has many
complete and heavily commented scripts demonstrating how to do that.


# Usage

All of the code in this repository is defined in a module, which is most easily used the way you
would use an official julia package. To faciliate that, I suggest adding the directory to your
LOAD_PATH, which can most easily be done by adding
```julia
push!(LOAD_PATH, "/path/to/the/src/directory/")
```
to your `.juliarc.jl` file. Alternatively, you could symlink this repository to your `~/.julia`
directory like this:
```
$ ln -s /path/to/this/repo/ /path/to/.julia/v0.6/KernelMatrices
```
Lately, though, I have gotten annoyed with some behavior this makes the package manager do. I suggest
the first alternative.

You will need to install a few packages that are listed in the REQUIRE file, which can easily be done
with 
```julia
julia> Pkg.add.(["StaticArrays", "IterTools", "GeometricalPredicates", "NearestNeighbors"])
```
None of those requirements are substantial or require any special care. If it all works, you will see
(at the time of writing, at least) the word `nothing` printed four times.

Once you have done those things, you can access the two modules---`KernelMatrices` and
`KernelMatrices.HODLR`---with
```julia
julia> using KernelMatrices, KernelMatrices.HODLR
```
<b> For both modules, nothing is exported to the namespace, so you will need to call every function
with `KernelMatrices.foo()`, or `HODLR.foo()`, or `KernelMatrices.HODLR.foo()` if you do not bring
the module `HODLR` into the namespace with `using KernelMatrices.HODLR`. </b>

<b> This code was written for Julia version `0.6.*`. </b>


# Changes to expect in the next release

1. <b> Updating the software to be compatible with julia `1.0.0` and beyond. </b> I have not done
   that now because I want to be careful to have an official release that is exactly the software I
   used for the corresponding submitted paper.

2. <b> Change the behavior of the level option. </b> As of now, giving a level of `0` gives what I
   consider to be the most sensible default behavior. But annoying, it means that giving a level of
   `k` really means your matrix will be of level `k-1` in the notation of the paper, so that a level
   of `1` will be an exact matrix. Again, I did not change this already because I only realized I had 
   written it in that awkward way after the official paper results had all been computed. But for 
   future releases, please expect different (and more reasonable) behavior for this arg.

3. <b> Small performance improvements </b>. In general, the code tries to be both fast and
   pedagogical. After this release, I am going to think more excluisvely about being fast.


# A note from the author

As mentioned above, I have made a serious effort to make the source code readable. Admittedly, some
of it---especially for the second derivatives of the kernel matrices---is challenging to parse. But
especially for the higher-level functions, like `HODLR.symmetricfactorize!`, I have made an effort
to name functions and their arguments in a helpful way, so I encourage any curious users to look at
the source directory.

If you would like to make contributions to the software, please feel free to reach out or submit
pull requests! It is my sincere hope that this software is actually useful to people, so I am
interested to hear about experiences messing with this software.

Please feel free to contact me at `cgeoga@anl.gov` with any comments, questions, or concerns. I
cannot promise that I will be in a position to help debug your code or implement the feature you
want, but I plan on being as supportive as possible of people who want to use this software, so it
is worth trying me.


# Citation

If you use this software suite, please cite [this paper](https://arxiv.org/abs/1808.03215).


# License

This software is distributed under the GNU GPL v2 license only. I am not interested in
redistributing it under any other license.

