
# As it stands now, this can't be used with Pkg.test("KernelMatrices"). But it 
# is still useful to run as a sanity check after any big changes are made.

# Load in the basic example matrix:
include("../example/basicexample.jl")
using KernelMatrices.HODLR, Test

BLAS.set_num_threads(1)

# Make sure this isn't run for a giant matrix that won't fit in memory:
N < 2^11 || error("This script is for testing on small matrices only.")

# Set a few method parameters:
mrnk = 72
lvl  = HODLR.LogLevel(8)
tvec = randn(N)
tmp1 = zeros(N)
tmp2 = zeros(N)

# Assemble the HODLR matrix, factorize it, get full for comparison:
HK   = HODLR.KernelHODLR(K, eps(), mrnk, lvl, nystrom=true, sorted=true)
HKf  = HODLR.full(HK)
HODLR.symmetricfactorize!(HK, verbose=false)
W    = HODLR.full(HK.W)

# Equality of symmetric factors:
@test isapprox(W*W', HKf)

# Equality of logdet:
@test isapprox(logdet(HKf), logdet(HK), rtol=1.0e-5)

# Equality of product:
@test isapprox(HKf*tvec, HK*tvec)

# Equality of solve:
@test isapprox(HKf\tvec, HK\tvec)

# Equality of factor solve:
ldiv!(tmp1, HK.W, tvec)
@test isapprox(tmp1, W\tvec)

# Equality of factor transpose solve:
ldiv!(tmp2, adjoint(HK.W), tmp1)
@test isapprox(tmp2, transpose(W)\tmp1)

