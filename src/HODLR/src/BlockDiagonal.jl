
# This is a copy of the external repository
# https://bitbucket.org/cgeoga/BlockDiagonal.jl.git.  Because making
# unregistered packages depend on other unregistered packages is an open
# question about design decision at the moment, I'm including the full source
# here and just absorbing the code. Not ideal, but I don't plan to change this
# code much, and it was a very simple dependency.

struct BDiagonal{T, MT} #<: AbstractMatrix{T}
  V  :: Vector{MT}
  Ix :: Vector{NTuple{4, Int64}} 
end

function BDiagonal(V)
  Ix = NTuple{4, Int64}[]
  for (j, Vj) in enumerate(V)
    j == 1 && push!(Ix, (1, size(Vj, 1), 1, size(Vj, 2)))
    (t1, t2) = Ix[end][2], Ix[end][4]
    j > 1  && push!(Ix, (t1 + 1, t1 + size(Vj, 1), t2 + 1, t2 + size(Vj, 2)))
  end
  BDiagonal{eltype(first(V)), eltype(V)}(V, Ix)
end

function Base.getindex(BD::BDiagonal{T, MT}, j::Int64, k::Int64) where{T, MT}
  for (ix, Vj) in zip(BD.Ix, BD.V)
    ix[1] <= j <= ix[2] && return (k < ix[3] || k > ix[4]) ? 0.0 : Vj[j-ix[1]+1, k-ix[3]+1]
  end
  throw(BoundsError(BD, (j,k)))
end

Base.size(BD::BDiagonal{T, MT}) where {T,MT} = (BD.Ix[end][2], BD.Ix[end][4])

for fn in (:factorize, :adjoint, :transpose, :inv)
  @eval $fn(BD::BDiagonal{T, MT}) where{T,MT} = BDiagonal($fn.(BD.V))
end

for (fn, rd) in zip((:tr, :det, :logdet), (:sum, :prod, :sum))
  @eval $fn(BD::BDiagonal{T, MT}) where{T,MT} = $rd($fn.(BD.V))
end

for (fn, ar) in Iterators.product((:ldiv!, :mul!), (:StridedVector, :StridedMatrix))
  @eval begin
    function $fn(target::$ar{T}, BD::BDiagonal{T, MT}, src::$ar{T}) where{T, MT}
      for (ix, Dj) in zip(BD.Ix, BD.V)
        @inbounds $fn(view(target, ix[1]:ix[2], :), Dj, view(src, ix[1]:ix[2], :))
      end
      target
    end
  end
  if fn == :ldiv!
    @eval begin
      function $fn(BD::BDiagonal{T, MT}, target::$ar{T}) where{T, MT}
        for (ix, Dj) in zip(BD.Ix, BD.V)
          @inbounds $fn(Dj, view(target, ix[1]:ix[2], :))
        end
        target
      end
    end
  end
end

function LinearAlgebra.:\(BD::BDiagonal{T, MT}, src::StridedArray{T}) where{T, MT}
  target = similar(src)
  MT <: Union{Factorization{T}, Adjoint{T,<:Factorization{T}}} && return ldiv!(target, BD, src)
  ldiv!(target, factorize(BD), src)
end

Base.:*(BD::BDiagonal{T, MT}, src::StridedArray{T}) where{T, MT} = mul!(similar(src), BD, src)


