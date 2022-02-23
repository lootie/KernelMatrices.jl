
@inline mul_t(A, B) = A*transpose(B)
@inline t_mul(A, B) = transpose(A)*B

function mapf(f::Function, v, plel::Bool, ptype::Symbol=:threaded)
  ex = (plel && Threads.nthreads() > 1) ? ThreadedEx() : SequentialEx()
  Folds.map(f, v, ex)
end

function foreachf(f::Function, v, plel::Bool, ptype::Symbol=:threaded)
  ex = (plel && Threads.nthreads() > 1) ? ThreadedEx() : SequentialEx()
  Folds.foreach(f, v, ex)
end

function saa_shuffle!(v::Vector{Vector{T}})::Nothing where{T<:Number}
  for j in eachindex(v)
    rand!(v[j], [-1.0, 1.0])
  end
  nothing
end

function givesaa(len::Int64, sz::Int64; seed::Int64=0)::Vector{Vector{Float64}}
  seed != 0 && Random.seed!(seed)
  vecs = map(x->Array{Float64}(undef, sz), 1:len)
  saa_shuffle!(vecs)
  return vecs
end

function lrsymfact(U12::Matrix{T}, U21::Matrix{T})::LowRankW{T} where{T<:Number}
  qr12  = qr!(U12)
  qr21  = qr!(U21)
  R1R2t = qr12.R*qr21.R'
  X     = cholesky!(Symmetric([I R1R2t; R1R2t' I])).L-I
  return LowRankW(cat(Matrix(qr12.Q), Matrix(qr21.Q), dims=[1,2]), X)
end

function lrx_solterm(W::LowRankW{T}, v::Array{T}) where{T<:Number}
  luf  = W.X + I 
  Xv   = W.X*v
  return Xv - W.X*(luf\Xv)
end

function lrx_solterm_t(W::LowRankW{T}, v::Array{T}) where{T<:Number}
  luf  = W.X' + I 
  Xv   = t_mul(W.X, v)
  return Xv - t_mul(W.X, (luf\Xv))
end

# If we could interleave the V and U matrices (in that order, so starting with
# V[1], then U[1], then V[2], ...) into a matrix M, then this function would
# just be BDiagonal(Wvec)\M. That's really what's going on here. But to avoid
# copies, I play a slightly more ornate game of breaking up the Wvec into the
# right sized pieces to apply to each Uj and Vj individually.
function invapply!(Wvec,
                   Uvec::Vector{Matrix{Float64}}, 
                   Vvec::Vector{Matrix{Float64}})
  stepsz = div(length(Wvec), 2*length(Uvec))
  ind    = collect(1:stepsz)
  for (Vj, Uj) in zip(Vvec, Uvec)
    ldiv!(BDiagonal(Wvec[ind]),         Vj) # @spawn?
    ldiv!(BDiagonal(Wvec[ind.+stepsz]), Uj) # @spawn?
    ind .+= 2*stepsz
  end
  nothing
end

function DBlock(K::KernelMatrix{T,N,A,Fn}, dfun::Function, lndmks::AbstractVector, 
                plel::Bool=false)::DerivativeBlock{T} where{T<:Number,N,A,Fn}
  K1p  = full(KernelMatrix(K.x1, lndmks, K.parms, K.kernel), plel)
  Kp2  = full(KernelMatrix(lndmks, K.x2, K.parms, K.kernel), plel)
  K1pd = full(KernelMatrix(K.x1, lndmks, K.parms, dfun), plel)
  Kp2d = full(KernelMatrix(lndmks, K.x2, K.parms, dfun), plel)
  return DerivativeBlock(K1p, K1pd, Kp2, Kp2d)
end

function DBlock_mul(B::DerivativeBlock{T}, src::Vector{T}, Kp::Cholesky{T, Matrix{T}},
                    Kpj::Symmetric{T, Matrix{T}})::Vector{T} where{T<:Number}
  out  = B.K1pd*(Kp\(B.Kp2*src))
  out -= B.K1p*(Kp\(Kpj*(Kp\(B.Kp2*src))))
  out += B.K1p*(Kp\(B.Kp2d*src))
  return out
end

function DBlock_mul_t(B::DerivativeBlock{T}, src::Vector{T}, Kp::Cholesky{T, Matrix{T}},
                      Kpj::Symmetric{T, Matrix{T}})::Vector{T} where{T<:Number}
  out  = t_mul(B.Kp2, (Kp\t_mul(B.K1pd, src)))
  out -= t_mul(B.Kp2, Kp\(Kpj*(Kp\(t_mul(B.K1p, src)))))
  out += t_mul(B.Kp2d, Kp\t_mul(B.K1p, src) )
  return out
end

function DBlock_full(B::DerivativeBlock{T}, Kp::Cholesky{T, Matrix{T}},
                     Kpj::Symmetric{T, Matrix{T}})::Matrix{T} where{T<:Number}
  Out  = B.K1pd*(Kp\B.Kp2)
  Out -= B.K1p*(Kp\(Kpj*(Kp\B.Kp2)))
  Out += B.K1p*(Kp\B.Kp2d)
  return Out
end

function SBlock(K::KernelMatrix{T,N,A,Fn}, djk::Function, lndmks::AbstractVector, 
                plel::Bool=false)::SecondDerivativeBlock{T} where{T<:Number,N,A,Fn}
  K1pdk = full(KernelMatrix(K.x1, lndmks, K.parms, djk), plel)
  Kp2dk = full(KernelMatrix(lndmks, K.x2, K.parms, djk), plel)
  return SecondDerivativeBlock(K1pdk, Kp2dk)
end

# The many, many parentheses are to ensure that things are done in the reasonable order.
# Like, for matrices A, B, and vector v, we definitely would want to do A*(B*v) instead of 
# (A*B)*v, for example. Unfortunately, the code looks all the more like an unreadable mess for it.
function SDBlock_mul(Bj::DerivativeBlock{T}, Bk::DerivativeBlock{T}, 
                     Bjk::SecondDerivativeBlock{T},
                     src::Vector{T}, Sp::Cholesky{T, Matrix{T}}, 
                     Spj::Symmetric{T,Matrix{T}},
                     Spk::Symmetric{T, Matrix{T}}, 
                     Spjk::Symmetric{T, Matrix{T}}) where{T<:Number}
  # The first term:
  out  = Bjk.K1pjk*(Sp\(Bj.Kp2*src))
  out -= Bj.K1pd*(Sp\(Spk*(Sp\(Bj.Kp2*src))))
  out += Bj.K1pd*(Sp\(Bk.Kp2d*src))
  # The second term:
  out += Bk.K1pd*(Sp\(Spj*(Sp\(Bj.Kp2*src))))
  out -= Bk.K1p*(Sp\(Spk*(Sp\(Spj*(Sp\(Bk.Kp2*src))))))
  out += Bk.K1p*(Sp\(Spjk*(Sp\(Bk.Kp2*src))))
  out -= Bk.K1p*(Sp\(Spj*(Sp\(Spk*(Sp\(Bk.Kp2*src))))))
  out += Bk.K1p*(Sp\(Spj*(Sp\(Bj.Kp2d*src))))
  # The third term:
  out += Bk.K1pd*(Sp\(Bk.Kp2*src))
  out -= Bk.K1p*(Sp\(Spk*(Sp\(Bk.Kp2*src))))
  out += Bk.K1p*(Sp\(Bjk.Kp2jk*src))
  # return it, and thank god that it is over:
  return out
end

function SDBlock_mul_t(Bj::DerivativeBlock{T}, Bk::DerivativeBlock{T},
                       Bjk::SecondDerivativeBlock{T}, src::Vector{T}, 
                       Sp::Cholesky{T,Matrix{T}},
                       Spj::Symmetric{T, Matrix{T}}, Spk::Symmetric{T, Matrix{T}},
                       Spjk::Symmetric{T, Matrix{T}}) where{T<:Number}
  # The first term:
  out  = t_mul(Bj.Kp2,  Sp\t_mul(Bjk.K1pjk, src))
  out -= t_mul(Bj.Kp2,  Sp\(Spk*(Sp\t_mul(Bj.K1pd, src))))
  out += t_mul(Bk.Kp2d, Sp\t_mul(Bk.K1pd, src))
  # The second term:
  out += t_mul(Bj.Kp2,  Sp\(Spj*(Sp\t_mul(Bk.K1pd, src))))
  out -= t_mul(Bj.Kp2,  Sp\(Spj*(Sp\(Spk*(Sp\t_mul(Bk.K1p, src))))))
  out += t_mul(Bj.Kp2,  Sp\(Spjk*(Sp\t_mul(Bk.K1p, src))))
  out -= t_mul(Bj.Kp2,  Sp\(Spk*(Sp\(Spj*(Sp\t_mul(Bk.K1p, src))))))
  out += t_mul(Bk.Kp2d, Sp\(Spj*(Sp\t_mul(Bk.K1p, src))))
  # The third term:
  out += t_mul(Bj.Kp2d,   Sp\t_mul(Bk.K1pd, src))
  out -= t_mul(Bj.Kp2d,   Sp\(Spk*(Sp\t_mul(Bk.K1p, src))))
  out += t_mul(Bjk.Kp2jk, Sp\t_mul(Bk.K1p, src))
  # return it, and thank god that it is over:
  return out
end

function Deriv2mul(DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                   D2B2::Vector{Vector{SecondDerivativeBlock{T}}}, 
                   D2BL::Vector{Symmetric{T,Matrix{T}}},
                   Sjk::Symmetric{T, Matrix{T}}, src::StridedVector) where{T<:Number}
  # Zero out the target vector:
  target = zeros(eltype(src), length(src))
  # Apply the leaves:
  for j in eachindex(DKj.L)
    c, d = DKj.leafindices[j][3:4]
    target[c:d] += D2BL[j]*src[c:d]
  end
  # Apply the non-leaves:
  for j in eachindex(DKj.B)
    for k in eachindex(DKj.B[j])
      a, b, c, d   = DKj.nonleafindices[j][k]
      target[c:d] += SDBlock_mul_t(DKj.B[j][k], DKk.B[j][k], D2B2[j][k], src[a:b], DKj.S, DKj.Sj, DKk.Sj, Sjk)
      target[a:b] += SDBlock_mul(  DKj.B[j][k], DKk.B[j][k], D2B2[j][k], src[c:d], DKj.S, DKj.Sj, DKk.Sj, Sjk)
    end
  end
  return target
end

# This will be SLOW:
function Deriv2full(DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                    D2B2::Vector{Vector{SecondDerivativeBlock{T}}}, 
                    D2BL::Vector{Symmetric{T,Matrix{T}}},
                    Sjk::Symmetric{T, Matrix{T}}, sz::Int64) where{T<:Number}
  out = zeros(T, sz, sz)
  for j in 1:sz
    tmp      = zeros(T, sz)
    tmp[j]   = one(T)
    out[:,j] = Deriv2mul(DKj, DKk, D2B2, D2BL, Sjk, tmp)
  end
  return out
end

