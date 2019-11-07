
module HODLR

  using  StaticArrays, KernelMatrices, SharedArrays
  using Statistics, LinearAlgebra, Distributed, Random

  import NearestNeighbors
  import KernelMatrices: KernelMatrix, submatrix, ACA, submatrix_nystrom, nlfisub, full, nystrom_uvt, NystromKernel
  import IterTools:      zip, chain, partition, drop, imap
  import LinearAlgebra:  factorize, mul!, ldiv!, logdet, det, adjoint, transpose, inv, tr

  include("./src/BlockDiagonal.jl")

  include("./src/structstypes.jl")
  
  include("./src/utils.jl")

  include("./src/baseoverloads.jl")

  include("./src/indices.jl")

  include("./src/constructor.jl")

  include("./src/symfact.jl")

  include("./src/stochasticgradient.jl")

  include("./src/stochastichessian.jl")

  include("./src/maxlikfunctions.jl")

  include("./src/optimization.jl")

  include("./src/exactmaxlik.jl")

end

