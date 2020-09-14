
# Here is an example of constructing a HODLR matrix
# in a matrix-free way using the KernelMatrix type.

println()
println("NOTE: if you are lazy but want see pre-compiled times, just run this script twice in the same REPL.")
println()

include("basicexample.jl")
using  KernelMatrices.HODLR

BLAS.set_num_threads(1)

# Assemble the HODLR matrix:
vrbs    = false              # Verbose flag:  Show timing?
plel    = true               # Parallel flag: Do the operation in parallel?
maxrank = 72                 # max rank:      Fix the maximum rank of the off-diagonal blocks? If no, give 0.
tol     = 1.0e-8             # tolerance:     If doing the ACA, terminate the partial factorization at this tol.
                             #                This argument is still necessary for nystrom=true, but
                             #                it doesn't do anything.
lvl     = HODLR.LogLevel(8)  # level:         The number of dydic splits of the matrix. If you want to fix it
                             #                at a certain level k, use HODLR.FixedLevel(k). This will not
                             #                scale with the complexity you want, though, so instead have the
                             #                level grow with log2(n) - k by providing HODLR.LogLevel(k).


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
@time HODLR.symmetricfactorize!(HK, verbose=true, plel=plel)

# Now the struct HK supports fast solves and log-determinants. Example syntax is:
  # HK \ randn(N)
  # logdet(HK)

# If you want the full symmetric factor W so that HK ≈ W*W', the syntax is:
  # full(HK.W)

