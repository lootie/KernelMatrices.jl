
module HODLR

  using  StaticArrays, KernelMatrices, LinearAlgebra, Distributed, Random
  import NearestNeighbors
  import KernelMatrices: KernelMatrix, submatrix, ACA, submatrix_nystrom, nlfisub
  import IterTools:      zip, chain, partition, drop, imap

  include("structstypes.jl")
  
  include("utils.jl")

  include("baseoverloads.jl")

  include("indices.jl")

  include("constructor.jl")

  include("symfact.jl")

  include("stochasticgradient.jl")

  include("stochastichessian.jl")

  include("maxlikfunctions.jl")

  include("optimization.jl")

end

