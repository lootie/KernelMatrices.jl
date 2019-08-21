
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

try 
  using BlockDiagonal
catch
  println("Installing small unregistered dependency BlockDiagonal.jl (https://bitbucket.org/cgeoga/blockdiagonal.jl")
  include("../deps/build.jl")
end

