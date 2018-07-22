
type IncompleteCholesky{T}
  L::Matrix{T}
end

type KernelMatrix{T}
  x1       ::AbstractVector
  x2       ::AbstractVector
  parms    ::AbstractVector{T}
  kernel   ::Function
end

type NystromKernel{T}  <: Function
  Kernel::Function
  parms::Vector{T}
  lndmk::AbstractVector
  tmp1 ::Vector{T}
  tmp2 ::Vector{T}
  F::Union{Base.LinAlg.Cholesky{T, Matrix{T}}, Base.LinAlg.BunchKaufman{T, Matrix{T}}}
end

function KernelMatrix{T<:Number}(pts1::AbstractVector, pts2::AbstractVector,
                                 parms::AbstractVector{T}, kern::Function)::KernelMatrix{T}
  if typeof(pts1[1]) != typeof(pts2[1])
    error("Your point vectors are not of the same type")
  end
  typ = typeof(kern(pts1[1], pts2[1], parms))
  return KernelMatrix{typ}(pts1, pts2, parms, kern)
end

function NystromKernel(T::Type, kern::Function, landmark::AbstractVector, 
                       parms::AbstractVector, ispd::Bool)::NystromKernel
  M   = zeros(T, length(landmark), length(landmark))
  @inbounds for j in eachindex(landmark)
    @inbounds for k in eachindex(landmark)
      M[j,k] = kern(landmark[j], landmark[k], parms)
    end
  end
  F = ispd ? cholfact!(Symmetric(M + 1.0e-12I)) : bkfact!(Symmetric(M))
  tmp1 = Array{T}(length(landmark))
  tmp2 = Array{T}(length(landmark))
  return NystromKernel{T}(kern, parms, landmark, tmp1, tmp2, F)
end

function (NK::NystromKernel{T}){T<:Number}(x::AbstractVector, y::AbstractVector, pr::Vector{T})
  pr == NK.parms || error("Parms don't match: need to re-generate nystrom matrix.")
  @inbounds for j in eachindex(NK.tmp1)
    NK.tmp1[j] = NK.Kernel(x, NK.lndmk[j], NK.parms)
    NK.tmp2[j] = NK.Kernel(NK.lndmk[j], y, NK.parms)
  end
  A_ldiv_B!(NK.F, NK.tmp2)
  return dot(NK.tmp1, NK.tmp2)
end

