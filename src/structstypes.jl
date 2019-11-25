
mutable struct KernelMatrix{T, N, A, Fn} 
  x1       ::AbstractVector{A}
  x2       ::AbstractVector{A}
  parms    ::SVector{N,T}
  kernel   ::Fn
end

function KernelMatrix(pts, pts2, prms::Vector, fn::Function)
  eltype(pts) == eltype(pts2) || error("Eltype of pts and pts2 must agree.")
  A = eltype(pts)
  T = eltype(prms)
  return KernelMatrix{T,length(prms),A,typeof(fn)}(pts, pts2, SVector{length(prms), T}(prms), fn)
end

mutable struct NystromKernel{T, N, A, Fn}  <: Function 
  Kernel::Fn
  parms::SVector{N,T}
  lndmk::Vector{A}
  F::Union{Cholesky{T, Matrix{T}}, BunchKaufman{T, Matrix{T}}}
end

function NystromKernel(kern::Function, landmark::AbstractVector, 
                       parms::AbstractVector, ispd::Bool)::NystromKernel
  T   = eltype(parms)
  M   = zeros(T, length(landmark), length(landmark))
  for j in eachindex(landmark)
    for k in eachindex(landmark)
      @inbounds M[j,k] = kern(landmark[j], landmark[k], parms)
    end
  end
  F = ispd ? cholesky!(Symmetric(M)) : bkfact!(Symmetric(M))
  return NystromKernel(kern, SVector{length(parms), Float64}(parms), landmark, F)
end

