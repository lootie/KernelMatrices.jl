
# Usage

All of the code in this repository is defined in a module, which is most easily used the way you
would use an official julia package. To faciliate that, I suggest adding the directory to your
LOAD_PATH, which can most easily be done by adding
```
push!(LOAD_PATH, "/path/to/the/src/directory/")
```
to your `.juliarc.jl` file. Alternatively, you could symlink this repository to your `~/.julia`
directory like this:
```
$ ln -s /path/to/this/repo/ /path/to/.julia/v0.6/KernelMatrices
```
(except without the period at the end), although I've lately gotten annoyed with some behavior this
makes the package manager do. I suggest the first alternative.

You'll need to install a few packages that are listed in the REQUIRE file, which can easily be done
with 
```
julia> Pkg.add.(["StaticArrays", "IterTools", "GeometricalPredicates", "NearestNeighbors"])
```
None of those requirements are substantial or require any special care. If it all works, you'll see
(at the time of writing, at least) the word `nothing` printed four times.

Once you've done those things, you can access the two modules---`KernelMatrices` and
`KernelMatrices.HODLR`---with
```
julia> using KernelMatrices, KernelMatrices.HODLR
```
For both modules, nothing is exported to the namespace, so you'll need to call every function with
`KernelMatrices.foo()`, or `HODLR.foo()`, or `KernelMatrices.HODLR.foo()` if you do not bring the
module `HODLR` into the namespace with `using KernelMatrices.HODLR`. 

This code was written for Julia version `0.6.*`. After the initial release, subsequent development
will include updating the code to work on Julia `0.7.*` and `1.0+`. I see no reason why those
updates would be substantial or detrimental to the performance of the software.


# Introduction

This is a software suite for working with matrices that are defined as
```
A[i,j] = F(x[i], y[j], v),
```
Where `x` and `y` are any kind of struct that allows getindex-style access (like a vector) and `v`
is a vector of parameters.  I have overloaded most (if not all) of the relevant Base operations for
the KernelMatrix struct, so that once you create the lightweight KernelMatrix struct, call it `K`,
with 
```
julia> K = KernelMatrices.KernelMatrix(xpts, ypts, kernel_parameters, kernel_function)
```
you can access the (i,j)-th element of that matrix with `K[i,j]`, the j-th column or row with
`K[:,j]` and `K[j,:]`, get the full matrix with `full(K)`, or even do matrix-vector operations with
`K*vec`, although I do not recommend doing that. Because `K` does not really store any part of the
matrix, it generates values on the fly, and so the matrix multiplication will be slow. I advise
using `K` only for getting individual entries or individual rows/columns.


# HODLR matrices

`basicexample.jl`, and then `HODLR.jl` in the ./examples directory will show you all that you need
to construct HODLR matrices. I've made a serious effort to overload many Base functions for arrays
for the `KernelMatrices.KernelHODLR struct`, so once you have your factorized HODLR matrix it should
be as easy as `HK*vec` for multiplication, `HK\vec` for solves, and `logdet(HK)` for
log-determinant.


# Final note

For general use, the two example files really show it all. So long as you have a kernel function and
some collection of points that implements `getindex`, you can make `KernelMatrix` and
`KernelMatrices.HODLR.KernelHODLR` objects. 

For specific applications to Gaussian processes, explore ./example/fitting/. The file
`test_fit_trustregion.jl` demonstrates a very verbose and well-commented setup of maximum
likelihohod estimation using a trust region method for optimization. Considering how straightforward
that code is and how low the boilerplate requirements are, I have not written exhaustive
documentation.

For the specific scripts used to generate the results in the associated paper with this software
package, explore ./example/paperscripts. I suggest looking at the `test_fit_*.jl` files in
./example/fitting/ for the most verbose and heavily commented examples of the boilerplate required.
But again, I have made a serious effort to make those scripts very readable, so hopefully what they
do is clear.

I have also made a serious effort to make the source code readable. Admittedly, some of
it---especially for the second derivative matrices---is challenging to parse. But especially for the
higher-level functions, I have made an effort to name functions and their arguments in a helpful
way, so I encourage any curious users to look at the source directory.


# License

This software is distributed under the GNU GPL v2 license only.

