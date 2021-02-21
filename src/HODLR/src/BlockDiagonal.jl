
#=
module BlockDiagonal

  export BDiagonal

  using  LinearAlgebra
  import LinearAlgebra: factorize, ldiv!, mul!, adjoint, transpose, inv, tr, det
  =#

  struct BDiagonal{T, MT} 
    issquare :: Bool
    V  :: Vector{MT}
    Ix :: Vector{NTuple{4, Int64}} 
  end

  function BDiagonal(V::Union{Vector{MT}, NTuple{N,MT}}) where{N,MT}
    Ix = NTuple{4, Int64}[]
    sizehint!(Ix, length(V))
    for (j, Vj) in enumerate(V)
      (s1, s2) = size(Vj)
      if (s1, s2) == (1,1) 
        @warn "1x1 blocks will introduce type instability in factorize." maxlog=1
      end
      j == 1 && push!(Ix, (1, s1, 1, s2))
      (t1, t2) = Ix[end][2], Ix[end][4]
      j > 1  && push!(Ix, (t1 + 1, t1 + s1, t2 + 1, t2 + s2))
    end
    issquare = mapreduce(x->size(x,1) == size(x,2), *, V)
    BDiagonal{eltype(first(V)), MT}(issquare, collect(V), Ix)
  end

  Base.eltype(BD::BDiagonal{T, MT}) where{T, MT} = T
  Base.size(BD::BDiagonal{T, MT}) where {T,MT} = (BD.Ix[end][2], BD.Ix[end][4])
  Base.size(BD::BDiagonal{T, MT}, j::Int64) where {T,MT} = size(BD)[j]

  # Singleton Checked factorize: since factorize(rand(1,1)) returns a Float64,
  # for some technical reasons about how the type is defined we need this
  # wrapper that keeps 1x1 matrices as matrices.
  sc_factorize(M) = size(M) == (1,1) ? M : factorize(M)
  factorize(BD::BDiagonal{T, MT}) where{T,MT} = BDiagonal(sc_factorize.(BD.V))

  for fn in (:adjoint, :transpose, :inv)
    @eval $fn(BD::BDiagonal{T, MT}) where{T,MT} = BDiagonal($fn.(BD.V))
  end

  for (fn, rd) in zip((:tr, :det, :logdet), (:sum, :prod, :sum))
    @eval $fn(BD::BDiagonal{T, MT}) where{T,MT} = $rd($fn, BD.V) 
  end

  for (fn, ar) in Iterators.product((:ldiv!, :mul!), (:StridedVector, :StridedMatrix))
    @eval begin
      function $fn(target::$ar{T}, BD::BDiagonal{T, MT}, src::$ar{T}) where{T, MT}
        for (ix, Dj) in zip(BD.Ix, BD.V)
          @inbounds $fn(view(target, ix[1]:ix[2], :), Dj, view(src, ix[3]:ix[4], :))
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

  function Base.:*(src::Adjoint{T,<:StridedArray{T}}, BD::BDiagonal{T,MT}) where{T,MT}
    adjoint((adjoint(BD)*src.parent))
  end

  function LinearAlgebra.:\(BD::BDiagonal{T, MT}, src::StridedArray{T}) where{T, MT}
    @assert(BD.issquare, "Matrix blocks must be square.")
    target = similar(src)
    if MT <: Union{Factorization{T}, Adjoint{T,<:Factorization{T}}} 
      return ldiv!(target, BD, src)
    end
    ldiv!(target, factorize(BD), src)
  end

  function Base.:*(BD::BDiagonal{T, MT}, src::StridedVector{T}) where{T, MT} 
    mul!(Array{T}(undef, size(BD, 1)), BD, src)
  end

  function Base.:*(BD::BDiagonal{T, MT}, src::StridedMatrix{T}) where{T, MT} 
    mul!(Array{T}(undef, size(BD, 1), size(src, 2)), BD, src)
  end

  # Check if B1 is a refinement of B2:
  function sizes_agree(B1, B2)
    ixs = UnitRange{Int64}[]
    length(B2.Ix) > length(B1.Ix) && return (false, ixs)
    ix = 1
    szs_1 = [x[4]-x[3]+1 for x in B1.Ix]
    for szj in (x[2]-x[1]+1 for x in B2.Ix)
      for k in ix:length(szs_1)
        if sum(szs_1[ix:k]) == szj 
          push!(ixs, ix:k)
          ix = k+1
          break
        end
        k == length(szs_1)+1 && return (false, ixs)
      end
    end
    return (true, ixs)
  end

  function ldiv!(B1::BDiagonal{T, MT1}, B2::BDiagonal{T, MT2}) where{T,MT1,MT2}
    size(B1)[2] == size(B2)[1] || throw(error("Sizes don't agree."))
    (agreement, ixs) = sizes_agree(B1, B2)
    @assert agreement "Block sizes don't agree in a way that is currently supported."
    for (j, ix) in enumerate(ixs)
      ldiv!(BDiagonal(B1.V[ix]), B2.V[j])
    end
    return B2
  end

  function LinearAlgebra.:\(B1::BDiagonal{T, MT1}, B2::BDiagonal{T, MT2}) where{T,MT1,MT2}
    target = deepcopy(B2)
    if MT1 <: Union{Factorization{T}, Adjoint{T,<:Factorization{T}}} 
      return ldiv!(B1, target)
    end
    ldiv!(factorize(B1), target)
  end

  # One could also implement mul! here, but since there is not an in-place mul!
  # as there is ldiv!, there's always going to be some allocation here, and for
  # my use cases I don't think more careful allocation would yield much speedup.
  function Base.:*(B1::BDiagonal{T, MT1}, B2::BDiagonal{T, MT2}) where{T,MT1,MT2}
    (agreement, ixs) = sizes_agree(B1, B2)
    @assert agreement "Block sizes don't agree in a way that is currently supported."
    prod_V = Vector{promote_type(MT1, MT2)}(undef, length(ixs))
    for (j, ix) in enumerate(ixs)
      prod_V[j] = BDiagonal(B1.V[ix])*B2.V[j]
    end
    return BDiagonal(prod_V)
  end

#end

