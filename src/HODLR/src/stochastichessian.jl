
# The solve term for the Hessian, corresponding to equation 27:
function HODLR_hess_slv(HK::KernelHODLR{T}, DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                        D2B2::Vector{Vector{SecondDerivativeBlock{T}}}, D2BL::Vector{Symmetric{T,Matrix{T}}},
                        Sjk::Symmetric{T, Matrix{T}}, data::Vector{T}, HK_s_data::Vector{T}) where{T<:Number}
  out  = -(HK\(DKk*(HK\(DKj*HK_s_data))))
  out -=  HK\(DKj*(HK\(DKk*HK_s_data)))
  out +=  HK\Deriv2mul(DKj, DKk, D2B2, D2BL, Sjk, HK_s_data)
  return dot(data, out)
end

function HODLR_p_hess_slv(HK::KernelHODLR{T}, DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                          D2B2::Vector{Vector{SecondDerivativeBlock{T}}}, D2BL::Vector{Symmetric{T,Matrix{T}}},
                          Sjk::Symmetric{T, Matrix{T}}, data::Vector{T}, HK_s_data::Vector{T},
                          dotm::T) where{T<:Number}
  tmp1 = DKj*HK_s_data
  tmp2 = DKk*HK_s_data
  out  = -(HK\(DKk*(HK\tmp1)))
  out -=  HK\(DKj*(HK\tmp2))
  out +=  HK\Deriv2mul(DKj, DKk, D2B2, D2BL, Sjk, HK_s_data)
  ret  = dot(data, out)/dotm
  t2   = dot(data, HK\tmp1)/abs2(dotm)
  t2  *= dot(data, HK\tmp2)
  ret += t2
  return length(data)*ret
end

function HODLR_hess_extra(K::KernelMatrix{T}, HK::KernelHODLR{T}, DKj::DerivativeHODLR{T},
                          DKk::DerivativeHODLR{T}, dfunjk::Function, data::Vector{T},
                          HK_s_data::Vector{T}, vecs::Vector{Vector{T}}; profile::Bool=false,
                          plel::Bool=false) where{T<:Number}
  D2Blocks = SecondDerivativeBlocks(K, dfunjk, HK.nonleafindices, HK.mrnk, plel)
  D2Leaves = SecondDerivativeLeaves(K, dfunjk, HK.leafindices, plel)
  lndmk    = K.x1[Int64.(round.(linspace(1, size(K)[1], HK.mrnk)))]
  Sjk      = Symmetric(full(KernelMatrices.KernelMatrix(lndmk, lndmk, K.parms, dfunjk)))
  slv      = ifelse(profile,
                    HODLR_p_hess_slv(HK, DKj, DKk, D2Blocks, D2Leaves, Sjk, data, HK_s_data, dot(data, HK_s_data)),
                    HODLR_hess_slv(  HK, DKj, DKk, D2Blocks, D2Leaves, Sjk, data, HK_s_data))
  t2       = mapreduce(v->HODLR_hess_tr2(HK,DKj,DKk,D2Blocks,D2Leaves,Sjk,v), +, vecs)/length(vecs)
  return slv, t2
end

# A special faster version for when the second derivative is zero:
function HODLR_hess_slv_no2d(HK::KernelHODLR{T}, DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                             data::Vector{T}, HK_s_data::Vector{T}) where{T<:Number}
  out  = -(HK\(DKk*(HK\(DKj*HK_s_data))))
  out -=  HK\(DKj*(HK\(DKk*HK_s_data)))
  return dot(data, out)
end

function HODLR_hess_tr1_sym_diag(HK::KernelHODLR{T}, DKj::DerivativeHODLR{T}, vec::Vector{T}) where{T<:Number}
  tp1 = Array{T}(length(vec))
  tp2 = Array{T}(length(vec))
  At_ldiv_B!(tp1, HK.W, vec)
  A_ldiv_B!(tp2, HK.W, DKj*tp1)
  return dot(tp2, tp2)
end

function HODLR_hess_tr1_sym_offdiag(HK::KernelHODLR{T}, DKj::DerivativeHODLR{T}, 
                                    DKk::DerivativeHODLR{T}, vec::Vector{T}) where{T<:Number}
  tp1 = Array{T}(length(vec))
  tp2 = Array{T}(length(vec))
  At_ldiv_B!(tp1, HK.W, vec)
  A_ldiv_B!(tp2, HK.W, DKj*tp1 + DKk*tp1)
  return dot(tp2, tp2)
end

function HODLR_hess_tr2(HK::KernelHODLR{T}, DKj::DerivativeHODLR{T}, DKk::DerivativeHODLR{T},
                        D2B2::Vector{Vector{SecondDerivativeBlock{T}}}, D2BL::Vector{Symmetric{T,Matrix{T}}},
                        Sjk::Symmetric{T, Matrix{T}}, vec::Vector{T}) where{T<:Number}
  tp1 = Array{T}(length(vec))
  At_ldiv_B!(tp1, HK.W, vec)
  return dot(tp1, Deriv2mul(DKj, DKk, D2B2, D2BL, Sjk, tp1))
end

function stoch_fisher(K::KernelMatrix{T}, HK::KernelHODLR{T}, DKs::Vector{DerivativeHODLR{T}},
                      vecs::Vector{Vector{T}}; plel::Bool=false, shuffle::Bool=false) where{T<:Number}
  # Some tests to make sure the call is legit:
  HK.U        == nothing         || error("The matrix needs to be factorized.")
  length(DKs) == length(K.parms) || error("You didn't supply the right number of gradient funs.")
  # Shuffle the SAA vecs if requested:
  if shuffle
    saa_shuffle!(vecs)
  end
  # Allocate the output:
  Out = zeros(length(DKs), length(DKs))
  # Fill in the diagonals first:
  for j in 1:length(DKs)
    Out[j,j] = 0.5*mapreduce(v->HODLR_hess_tr1_sym_diag(HK, DKs[j], v), +, vecs)/length(vecs)
  end
  # Fill in the off-diagonals:
  for j in 1:length(DKs)
    for k in (j+1):length(DKs)
      tmp1     = 0.5*mapreduce(v->HODLR_hess_tr1_sym_offdiag(HK, DKs[j], DKs[k],  v), +, vecs)/length(vecs)
      Out[j,k] = 0.5*(tmp1 - Out[j,j] - Out[k,k])
    end
  end
  return Symmetric(Out)
end

function stoch_hessian(K::KernelMatrix{T}, HK::KernelHODLR{T}, dat::Vector{T},
                      d1funs::Vector{Function}, d2funs::Vector{Vector{Function}},
                      vecs::Vector{Vector{T}}; plel::Bool=false, verbose::Bool=false,
                      shuffle::Bool=false, profile::Bool=false) where{T<:Number}
  # Some tests to make sure the call is legit:
  d2lentest = Int64(length(d1funs)*(length(d1funs)+1)/2)
  HK.U           == nothing         || error("The matrix needs to be factorized.")
  length(d1funs) == length(K.parms) || error("You didn't supply the right number of gradient funs.")
  sum(length.(d2funs)) == d2lentest || error("You didn't supply the right number of Hessian funs.")
  # Do this solve once:
  HKsd = HK\dat 
  # Get the derivative functions:
  DKs = map(df -> DerivativeHODLR(K, df, HK, plel=plel), d1funs)
  Out = -full(stoch_fisher(K, HK, DKs, vecs, plel=plel, shuffle=shuffle))
  # Now add the things for the Hessian beyond the fisher matrix:
  for j in 1:length(d1funs)
    for k in j:length(d1funs)
      dfunjk = d2funs[j][k-j+1]
      if dfunjk != ZeroFunction()
        sterm,t2 = HODLR_hess_extra(K, HK, DKs[j], DKs[k], dfunjk, dat, HKsd, vecs, 
                                    plel=plel, profile=profile)
        Out[j,k]+= (0.5*t2 - 0.5*sterm)
      else
        profile && error("Oof, I didn't implement this speed-up for the profile likelihood.")
        sterm    = HODLR_hess_slv_no2d(HK, DKs[j], DKs[k], dat, HKsd)
        Out[j,k]-= 0.5*sterm
      end
    end
  end
  return Symmetric(Out)
end

