
module KernelMatrices

  using StaticArrays, LinearAlgebra
  import IterTools
  import LinearAlgebra: mul!

  include("structstypes.jl")

  include("baseoverloads.jl")

  include("utils.jl")

  include("factorizations.jl")

  # Its own module:
  include("HODLR/HODLR.jl")

end

