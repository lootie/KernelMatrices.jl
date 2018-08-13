
# Here is an example of constructing a HODLR matrix
# in a matrix-free way using the KernelMatrix type.

include("basicexample.jl")
using  KernelMatrices.HODLR

# Assemble the HODLR matrix:
vrbs    = false    # Verbose flag:  Show timing?
plel    = false    # Parallel flag: Do the operation in parallel?
maxrank = 72       # max rank:      Fix the maximum rank of the off-diagonal blocks? If no, give 0.
tol     = 1.0e-8   # tolerance:     If doing the ACA, terminate the partial factorization at this tol.
lvl     = 0        # level:         The number of dydic splits of the matrix. 0 leads to the default
                   #                value of log2(n) - 8. Due to how this is presently coded,
                   #                though, if you were going to supply it by hand, you should give
                   #                it log2(n) - 7.
                   #                !!!
                   #                IN GENERAL, IF YOU WANT TO SPECIFY A LEVEL K, PLEASE SUPPLY K+1
                   #                TO THIS ARGUMENT.  
                   #                !!!
                   #                So, supplying the number 1 will give you an exact matrix,
                   #                corresponding to a level 0. In a future release that isn't tied
                   #                to a submitted paper, I will fix this behavior to be more
                   #                intuitive.



# Here are sample function calls to the constructor HODLR.KernelHODLR which showcase the different
# options. If you want to do a solve or compute the logdet of the matrix, you must factorize it
# first. For the factorization to be possible, the matrix MUST be positive definite. If you use
# nystrom=true, this is guaranteed to be true if the original K is positive definite. If you use the
# ACA, there is no such guarantee. But if your matrix is sufficiently well-behaved, it could work out.

println()
println("Not-precompiled assembly of HODLR matrix with ACA blocks, N=$N and ɛ=$tol is done in:")
@time HK = HODLR.KernelHODLR(K, tol, 0,       lvl, nystrom=false, plel=plel) ;

println()
println("Not-precompiled assembly of HODLR matrix with fixed-rank ACA blocks, N=$N and p=$maxrank is done in:")
@time HK = HODLR.KernelHODLR(K, tol, maxrank, lvl, nystrom=false, plel=plel) ;

println()
println("Not-precompiled assembly of HODLR matrix with Nystrom blocks, N=$N and p=$maxrank is done in:")
@time HK = HODLR.KernelHODLR(K, tol, maxrank, lvl, nystrom=true,  plel=plel) ;

println()
println("Not-precommpiled symmetric factorization of the last matrix is done in:")
@time HODLR.symmetricfactorize!(HK, verbose=vrbs)

# Now the struct HK supports fast solves and log-determinants. Example syntax is:
  # HK \ randn(N)
  # logdet(HK)

# If you want the full symmetric factor W so that HK ≈ W*W', the syntax is:
  # full(HK.W)

