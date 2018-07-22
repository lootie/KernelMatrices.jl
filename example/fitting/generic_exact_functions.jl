
# These need to be in scope for this script to be loaded:
using KernelMatrices, KernelMatrices.HODLR

function exact_nll_objective(prms::AbstractVector, grad::Vector, locs::AbstractVector,
                             dats::AbstractVector, kernfun::Function, dfuns::Vector{Function}, vrb::Bool)
  vrb && println(prms)
  t1 = @elapsed begin
  K   = cholfact!(full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  nll = 0.5*logdet(K) + 0.5*dot(dats, K\dats)
  end
  vrb && println("Cholesky + solve + logdet took $t1 seconds.")
  t2 = @elapsed begin
  if length(grad) > 0
    grad .= exact_gradient(prms, locs, dats, kernfun, dfuns)
  end
  end
  vrb && println("The exact gradient took        $t2 seconds.")
  return nll
end

function exact_nlpl_objective(prms::AbstractVector, grad::Vector, locs::AbstractVector,
                             dats::AbstractVector, kernfun::Function, dfuns::Vector{Function}, vrb::Bool)
  vrb && println(prms)
  t1 = @elapsed begin
  K   = cholfact!(full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  nll = 0.5*logdet(K) + 0.5*length(dats)*log(dot(dats, K\dats))
  end
  vrb && println("Cholesky + solve + logdet took $t1 seconds.")
  t2 = @elapsed begin
  if length(grad) > 0
    grad .= exact_profile_gradient(prms, locs, dats, kernfun, dfuns)
  end
  end
  vrb && println("The exact gradient took        $t2 seconds.")
  return nll
end

function exact_nlpl_scale(prms::AbstractVector, locs::AbstractVector,
                          dats::AbstractVector, kernfun::Function)
  K   = cholfact!(full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = dot(dats, K\dats)/length(dats)
end

function exact_gradient_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                             dats::AbstractVector, K::Base.LinAlg.Cholesky{Float64,Matrix{Float64}},
                             drfun::Function)::Float64
  dK  = full(KernelMatrices.KernelMatrix(locs, locs, prms, drfun))
  out = 0.5*trace(K\dK) - 0.5*dot(dats, K\(dK*(K\dats)))
  return out
end

function exact_gradient(prms::AbstractVector{Float64}, locs::AbstractVector, dats::AbstractVector,
                        kernfun::Function, dfuns::Vector{Function})
  K   = cholfact!(full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = zeros(length(dfuns))
  if nworkers() > 1
    out = pmap(dfj->exact_gradient_term(prms, locs, dats, K, dfj), dfuns)
  else
    for j in eachindex(out)
      out[j] = exact_gradient_term(prms, locs, dats, K, dfuns[j])
    end
  end
  return out
end

function exact_HODLR_gradient_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                                   dats::AbstractVector, K::KernelMatrices.KernelMatrix{Float64},
                                   HK::HODLR.KernelHODLR{Float64},
                                   HKf::Base.LinAlg.Cholesky{Float64,Matrix{Float64}},
                                   drfun::Function, plel::Bool=false)::Float64
  dK  = full(HODLR.DerivativeHODLR(K, drfun, HK, plel=plel))
  return 0.5*(trace(HKf\dK) - dot(dats, HKf\(dK*(HKf\dats))))
end

function exact_HODLR_gradient(prms::AbstractVector{Float64}, locs::AbstractVector,
                              dats::AbstractVector, opts::HODLR.Maxlikopts)
  nK  = KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)
  HK  = HODLR.KernelHODLR(nK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  HKf = cholfact!(full(HK))
  out = zeros(length(opts.dfuns))
  for j in eachindex(out)
    out[j] = exact_HODLR_gradient_term(prms, locs, dats, nK, HK, HKf, opts.dfuns[j], opts.apll)
  end
  return out
end

function exact_p_gradient_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                               dats::AbstractVector, K::Base.LinAlg.Cholesky{Float64,Matrix{Float64}},
                               drfun::Function)::Float64
  dK  = full(KernelMatrices.KernelMatrix(locs, locs, prms, drfun))
  Ks  = K\dats
  out = 0.5*trace(K\dK) - 0.5*length(dats)*dot(dats, K\(dK*(K\dats)))/dot(dats, Ks)
  return out
end

function exact_profile_gradient(prms::AbstractVector{Float64}, locs::AbstractVector, dats::AbstractVector,
                                kernfun::Function, dfuns::Vector{Function})
  K   = cholfact!(full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = zeros(length(dfuns))
  if nworkers() > 1
    out = pmap(dfj->exact_p_gradient_term(prms, locs, dats, K, dfj), dfuns)
  else
    for j in eachindex(out)
      out[j] = exact_p_gradient_term(prms, locs, dats, K, dfuns[j])
    end
  end
  return out
end

function exact_hessian_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                            dats::AbstractVector, K::Base.LinAlg.Cholesky{Float64, Matrix{Float64}}, 
                            dKj::Matrix{Float64}, drfunk::Function, drfunjk::Function)::Float64
  if drfunjk != HODLR.ZeroFunction()
    # Get all the required derivative matrices in place:
    dKk   = full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunk))
    dKjk  = full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunjk))
    # Compute the solve term:
    o_sv  = dot(dats, K\(dKjk*(K\dats)))
    o_sv -= dot(dats, K\(dKk*(K\(dKj*(K\dats)))))
    o_sv -= dot(dats, K\(dKj*(K\(dKk*(K\dats)))))
    # Compute the trace term:
    o_tr  = trace(K\dKjk)
    o_tr -= trace(K\(dKk*(K\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  else
    # Get all the required derivative matrices in place:
    dKk   = full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunk))
    # Compute the solve term:
    o_sv  = -dot(dats, K\(dKk*(K\(dKj*(K\dats)))))
    o_sv -= dot(dats, K\(dKj*(K\(dKk*(K\dats)))))
    # Compute the trace term:
    o_tr  = -trace(K\(dKk*(K\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  end
end

function exact_HODLR_hessian_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                                  dats::AbstractVector, opts::HODLR.Maxlikopts,
                                  K::KernelMatrices.KernelMatrix{Float64},
                                  HK::HODLR.KernelHODLR{Float64} ,
                                  HKf::Base.LinAlg.Cholesky{Float64, Matrix{Float64}},
                                  dKj_::HODLR.DerivativeHODLR{Float64}, 
                                  dKj::Matrix{Float64},
                                  drfunk::Function,
                                  drfunjk::Function)::Float64
  if drfunjk != HODLR.ZeroFunction()
    # Get all the required derivative matrices in place:
    dKk_  = HODLR.DerivativeHODLR(K, drfunk, HK, plel=opts.apll)
    dKk   = full(dKk_)
    D2B   = HODLR.SecondDerivativeBlocks(K, drfunjk, HK.nonleafindices, HK.mrnk, opts.apll)
    D2L   = HODLR.SecondDerivativeLeaves(K, drfunjk, HK.leafindices, opts.apll)
    lmk   = K.x1[Int64.(round.(linspace(1, size(K)[1], HK.mrnk)))]
    Sjk   = Symmetric(full(KernelMatrices.KernelMatrix(lmk, lmk, K.parms, drfunjk)))
    dKjk  = HODLR.Deriv2full(dKj_, dKk_, D2B, D2L, Sjk, length(locs))
    # Compute the solve term:
    o_sv  = dot(dats, HKf\(dKjk*(HKf\dats)))
    o_sv -= dot(dats, HKf\(dKk*(HKf\(dKj*(HKf\dats)))))
    o_sv -= dot(dats, HKf\(dKj*(HKf\(dKk*(HKf\dats)))))
    # Compute the trace term:
    o_tr  = trace(HKf\dKjk)
    o_tr -= trace(HKf\(dKk*(HKf\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  else
    # Get all the required derivative matrices in place:
    dKk_  = HODLR.DerivativeHODLR(K, drfunk, HK, plel=opts.apll)
    dKk   = full(dKk_)
    # Compute the solve term:
    o_sv  = -dot(dats, HKf\(dKk*(HKf\(dKj*(HKf\dats)))))
    o_sv -= dot(dats, HKf\(dKj*(HKf\(dKk*(HKf\dats)))))
    # Compute the trace term:
    o_tr  = -trace(HKf\(dKk*(HKf\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  end
end

function exact_hessian(prms::AbstractVector,locs::AbstractVector, dats::AbstractVector,
                       kernfun::Function, dfuns::Vector{Function}, d2funs::Vector{Vector{Function}})
  K   = cholfact!(full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj = full(KernelMatrices.KernelMatrix(locs, locs, prms, dfuns[j]))
    for k in j:length(dfuns)
      out[j,k] = exact_hessian_term(prms, locs, dats, K, dKj, dfuns[k], d2funs[j][k-j+1])
    end
  end
  return Symmetric(out)
end

function exact_HODLR_hessian(prms::AbstractVector,locs::AbstractVector, dats::AbstractVector,
                             opts::HODLR.Maxlikopts, d2funs::Vector{Vector{Function}})
  nK  = KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)
  HK  = HODLR.KernelHODLR(nK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  HKf = cholfact!(full(HK))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj  = HODLR.DerivativeHODLR(nK, opts.dfuns[j], HK, plel=opts.apll)
    dKjf = full(dKj)
    for k in j:length(dfuns)
      out[j,k] = exact_HODLR_hessian_term(prms,locs,dats,opts,nK,HK,HKf,dKj,dKjf,dfuns[k],d2funs[j][k-j+1])
    end
  end
  return Symmetric(out)
end

function exact_HODLR_fisher(prms::AbstractVector,locs::AbstractVector, dats::AbstractVector,
                            opts::HODLR.Maxlikopts)
  nK  = KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)
  HK  = HODLR.KernelHODLR(nK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  HKf = cholfact!(full(HK))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj  = HODLR.DerivativeHODLR(nK, opts.dfuns[j], HK, plel=opts.apll)
    dKjf = full(dKj)
    for k in j:length(dfuns)
      dKk  = HODLR.DerivativeHODLR(nK, opts.dfuns[k], HK, plel=opts.apll)
      dKkf = full(dKk)
      out[j,k] = 0.5*trace(HKf\(dKjf*(HKf\dKkf)))
    end
  end
  return Symmetric(out)
end

function exact_fisher_matrix(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector,
                             kernfun::Function, dfuns::Vector{Function})
  K   = cholfact!(full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj = full(KernelMatrices.KernelMatrix(locs, locs, prms, dfuns[j]))
    for k in j:length(dfuns)
      dKk      = full(KernelMatrices.KernelMatrix(locs, locs, prms, dfuns[k]))
      out[j,k] = 0.5*trace(K\(dKj*(K\dKk)))
    end
  end
  return Symmetric(out)
end

