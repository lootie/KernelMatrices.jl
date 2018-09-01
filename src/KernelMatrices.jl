
module KernelMatrices

  using StaticArrays, LinearAlgebra
  import IterTools, GeometricalPredicates

  include("structstypes.jl")

  include("baseoverloads.jl")

  include("utils.jl")

  include("factorizations.jl")

  # Its own module:
  include("HODLR/src/HODLR.jl")

end

