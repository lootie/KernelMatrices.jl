
module HODLR

  using  StaticArrays, KernelMatrices, LinearAlgebra, Distributed, Random
  import NearestNeighbors
  import KernelMatrices: KernelMatrix, submatrix, ACA, submatrix_nystrom, nlfisub, full
  import IterTools:      zip, chain, partition, drop, imap
  import LinearAlgebra:  mul!, ldiv!, logdet, det

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

