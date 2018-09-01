
mutable struct IncompleteCholesky{T<:Number} 
  L::Matrix{T}
end

mutable struct KernelMatrix{T<:Number, A} 
  x1       ::AbstractVector{A}
  x2       ::AbstractVector{A}
  parms    ::AbstractVector{T}
  kernel   ::Function
end

mutable struct NystromKernel{T<:Number}  <: Function 
  Kernel::Function
  parms::Vector{T}
  lndmk::AbstractVector
  tmp1 ::Vector{T}
  tmp2 ::Vector{T}
  F::Union{Cholesky{T, Matrix{T}}, BunchKaufman{T, Matrix{T}}}
end

function NystromKernel(T::Type, kern::Function, landmark::AbstractVector, 
                       parms::AbstractVector, ispd::Bool)::NystromKernel
  M   = zeros(T, length(landmark), length(landmark))
  @inbounds for j in eachindex(landmark)
    @inbounds for k in eachindex(landmark)
      M[j,k] = kern(landmark[j], landmark[k], parms)
    end
  end
  F = ispd ? cholesky!(Symmetric(M + 1.0e-12I)) : bkfact!(Symmetric(M))
  tmp1 = Array{T}(undef, length(landmark))
  tmp2 = Array{T}(undef, length(landmark))
  return NystromKernel{T}(kern, parms, landmark, tmp1, tmp2, F)
end

function (NK::NystromKernel{T})(x::A, y::A, pr::Vector{T}) where{T<:Number, A<:AbstractVector} 
  pr == NK.parms || error("Parms don't match: need to re-generate nystrom matrix.")
  @inbounds for j in eachindex(NK.tmp1)
    NK.tmp1[j] = NK.Kernel(x, NK.lndmk[j], NK.parms)
    NK.tmp2[j] = NK.Kernel(NK.lndmk[j], y, NK.parms)
  end
  A_ldiv_B!(NK.F, NK.tmp2)
  return dot(NK.tmp1, NK.tmp2)
end

