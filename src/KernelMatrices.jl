
module KernelMatrices

  using StaticArrays, SpecialFunctions, SharedArrays, Distributed, LinearAlgebra

  import IterTools
  import LinearAlgebra: mul!

  include("structstypes.jl")

  include("baseoverloads.jl")

  include("utils.jl")

  include("factorizations.jl")

  include("covariancefunctions.jl")

  # Its own module:
  include("HODLR/HODLR.jl")

end

