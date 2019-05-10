
function exact_nll_objective(prms::AbstractVector, grad::Vector, locs::AbstractVector,
                             dats::AbstractVector, kernfun::Function, dfuns::Vector{Function}, vrb::Bool)
  vrb && println(prms)
  t1 = @elapsed begin
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
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
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
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
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = dot(dats, K\dats)/length(dats)
end

function exact_gradient_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                             dats::AbstractVector, K::LinearAlgebra.Cholesky{Float64,Matrix{Float64}},
                             drfun::Function)::Float64
  dK  = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, drfun))
  out = 0.5*tr(K\dK) - 0.5*dot(dats, K\(dK*(K\dats)))
  return out
end

function exact_gradient(prms::AbstractVector{Float64}, locs::AbstractVector, dats::AbstractVector,
                        kernfun::Function, dfuns::Vector{Function})
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
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
                                   HKf::LinearAlgebra.Cholesky{Float64,Matrix{Float64}},
                                   drfun::Function, plel::Bool=false)::Float64
  dK  = KernelMatrices.full(HODLR.DerivativeHODLR(K, drfun, HK, plel=plel))
  return 0.5*(tr(HKf\dK) - dot(dats, HKf\(dK*(HKf\dats))))
end

function exact_HODLR_gradient(prms::AbstractVector{Float64}, locs::AbstractVector,
                              dats::AbstractVector, opts::HODLR.Maxlikopts)
  nK  = KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)
  HK  = HODLR.KernelHODLR(nK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  HKf = cholesky!(KernelMatrices.full(HK))
  out = zeros(length(opts.dfuns))
  for j in eachindex(out)
    out[j] = exact_HODLR_gradient_term(prms, locs, dats, nK, HK, HKf, opts.dfuns[j], opts.apll)
  end
  return out
end

function exact_HODLR_nll_objective(prms::AbstractVector{Float64}, g::AbstractVector,
                                   locs::AbstractVector, dats::AbstractVector, opts::HODLR.Maxlikopts)
  nK  = KernelMatrices.KernelMatrix(locs, locs, prms, opts.kernfun)
  HK  = HODLR.full(HODLR.KernelHODLR(nK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll))
  if length(g) > 0
    g .= exact_HODLR_gradient(prms, locs, dats, opts)
  end
  return 0.5*logdet(HK) + 0.5*dot(dats, HK\dats)
end

function exact_p_gradient_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                               dats::AbstractVector, K::LinearAlgebra.Cholesky{Float64,Matrix{Float64}},
                               drfun::Function)::Float64
  dK  = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, drfun))
  Ks  = K\dats
  out = 0.5*tr(K\dK) - 0.5*length(dats)*dot(dats, K\(dK*(K\dats)))/dot(dats, Ks)
  return out
end

function exact_profile_gradient(prms::AbstractVector{Float64}, locs::AbstractVector, dats::AbstractVector,
                                kernfun::Function, dfuns::Vector{Function})
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
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
                            dats::AbstractVector, K::LinearAlgebra.Cholesky{Float64, Matrix{Float64}}, 
                            dKj::Matrix{Float64}, drfunk::Function, drfunjk::Function)::Float64
  if !(typeof(drfunjk) == HODLR.ZeroFunction)
    # Get all the required derivative matrices in place:
    dKk   = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunk))
    dKjk  = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunjk))
    # Compute the solve term:
    o_sv  = dot(dats, K\(dKjk*(K\dats)))
    o_sv -= dot(dats, K\(dKk*(K\(dKj*(K\dats)))))
    o_sv -= dot(dats, K\(dKj*(K\(dKk*(K\dats)))))
    # Compute the tr term:
    o_tr  = tr(K\dKjk)
    o_tr -= tr(K\(dKk*(K\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  else
    # Get all the required derivative matrices in place:
    dKk   = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunk))
    # Compute the solve term:
    o_sv  = -dot(dats, K\(dKk*(K\(dKj*(K\dats)))))
    o_sv -= dot(dats, K\(dKj*(K\(dKk*(K\dats)))))
    # Compute the trace term:
    o_tr  = -tr(K\(dKk*(K\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  end
end


function exact_p_hessian_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                              dats::AbstractVector, K::LinearAlgebra.Cholesky{Float64, Matrix{Float64}}, 
                              dKj::Matrix{Float64}, drfunk::Function, drfunjk::Function)::Float64
  # Get all the required derivative matrices in place:
  dKk   = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunk))
  dKjk  = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, drfunjk))
  # Get the trace term:
  trt   = tr(K\dKjk)
  trt  -= tr(K\dKk*(K\dKj))
  # Get the solve term:
  tmp1  = dot(dats, K\dats)
  tmpj  = dot(dats, K\(dKj*(K\dats)))
  tmpk  = dot(dats, K\(dKk*(K\dats)))
  tmpjk = dot(dats, K\(dKk*(K\(dKj*(K\dats)))))
  tmpjk-= dot(dats, K\(dKjk*(K\dats)))
  tmpjk+= dot(dats, K\(dKj*(K\(dKk*(K\dats)))))
  # Return the term:
  return 0.5*trt + 0.5*length(dats)*(tmpjk/tmp1 - tmpj*tmpk/abs2(tmp1))
end


function exact_HODLR_hessian_term(prms::AbstractVector{Float64}, locs::AbstractVector,
                                  dats::AbstractVector, opts::HODLR.Maxlikopts,
                                  K::KernelMatrices.KernelMatrix{Float64},
                                  HK::HODLR.KernelHODLR{Float64} ,
                                  HKf::LinearAlgebra.Cholesky{Float64, Matrix{Float64}},
                                  dKj_::HODLR.DerivativeHODLR{Float64}, 
                                  dKj::Matrix{Float64},
                                  drfunk::Function,
                                  drfunjk::Function)::Float64
  if !(typeof(drfunjk) == HODLR.ZeroFunction)
    # Get all the required derivative matrices in place:
    dKk_  = HODLR.DerivativeHODLR(K, drfunk, HK, plel=opts.apll)
    dKk   = HODLR.full(dKk_)
    D2B   = HODLR.SecondDerivativeBlocks(K, drfunjk, HK.nonleafindices, HK.mrnk, opts.apll)
    D2L   = HODLR.SecondDerivativeLeaves(K, drfunjk, HK.leafindices, opts.apll)
    lmk   = K.x1[Int64.(round.(linspace(1, size(K)[1], HK.mrnk)))]
    Sjk   = Symmetric(KernelMatrices.full(KernelMatrices.KernelMatrix(lmk, lmk, K.parms, drfunjk)))
    dKjk  = HODLR.Deriv2full(dKj_, dKk_, D2B, D2L, Sjk, length(locs))
    # Compute the solve term:
    o_sv  = dot(dats, HKf\(dKjk*(HKf\dats)))
    o_sv -= dot(dats, HKf\(dKk*(HKf\(dKj*(HKf\dats)))))
    o_sv -= dot(dats, HKf\(dKj*(HKf\(dKk*(HKf\dats)))))
    # Compute the trace term:
    o_tr  = tr(HKf\dKjk)
    o_tr -= tr(HKf\(dKk*(HKf\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  else
    # Get all the required derivative matrices in place:
    dKk_  = HODLR.DerivativeHODLR(K, drfunk, HK, plel=opts.apll)
    dKk   = HODLR.full(dKk_)
    # Compute the solve term:
    o_sv  = -dot(dats, HKf\(dKk*(HKf\(dKj*(HKf\dats)))))
    o_sv -= dot(dats, HKf\(dKj*(HKf\(dKk*(HKf\dats)))))
    # Compute the trace term:
    o_tr  = -tr(HKf\(dKk*(HKf\dKj)))
    # return the term:
    return 0.5*o_tr - 0.5*o_sv
  end
end


function exact_hessian(prms::AbstractVector,locs::AbstractVector, dats::AbstractVector,
                       kernfun::Function, dfuns::Vector{Function}, d2funs::Vector{Vector{Function}})
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, dfuns[j]))
    for k in j:length(dfuns)
      out[j,k] = exact_hessian_term(prms, locs, dats, K, dKj, dfuns[k], d2funs[j][k-j+1])
    end
  end
  return Symmetric(out)
end


function exact_p_hessian(prms::AbstractVector,locs::AbstractVector, dats::AbstractVector,
                         kernfun::Function, dfuns::Vector{Function}, d2funs::Vector{Vector{Function}})
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, dfuns[j]))
    for k in j:length(dfuns)
      out[j,k] = exact_p_hessian_term(prms, locs, dats, K, dKj, dfuns[k], d2funs[j][k-j+1])
    end
  end
  return Symmetric(out)
end


function exact_HODLR_hessian(prms::AbstractVector,locs::AbstractVector, dats::AbstractVector,
                             opts::HODLR.Maxlikopts, d2funs::Vector{Vector{Function}})
  nK  = KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)
  HK  = HODLR.KernelHODLR(nK, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  HKf = cholesky!(HODLR.full(HK))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj  = HODLR.DerivativeHODLR(nK, opts.dfuns[j], HK, plel=opts.apll)
    dKjf = HODLR.full(dKj)
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
  HKf = cholesky!(HODLR.full(HK))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj  = HODLR.DerivativeHODLR(nK, opts.dfuns[j], HK, plel=opts.apll)
    dKjf = HODLR.full(dKj)
    for k in j:length(dfuns)
      dKk  = HODLR.DerivativeHODLR(nK, opts.dfuns[k], HK, plel=opts.apll)
      dKkf = HODLR.full(dKk)
      out[j,k] = 0.5*tr(HKf\(dKjf*(HKf\dKkf)))
    end
  end
  return Symmetric(out)
end

function exact_fisher_matrix(prms::AbstractVector, locs::AbstractVector, dats::AbstractVector,
                             kernfun::Function, dfuns::Vector{Function})
  K   = cholesky!(KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, kernfun)))
  out = zeros(length(dfuns), length(dfuns))
  for j in eachindex(dfuns)
    dKj = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, dfuns[j]))
    for k in j:length(dfuns)
      dKk      = KernelMatrices.full(KernelMatrices.KernelMatrix(locs, locs, prms, dfuns[k]))
      out[j,k] = 0.5*tr(K\(dKj*(K\dKk)))
    end
  end
  return Symmetric(out)
end

# Basically the same as the hierarchically accelerated version in optimization.jl, except this calls
# all the exact methods here. I really should write a generic version of the function at some point
# so I don't need two copies that are basically the same, but this is really just here for
# troubleshooting and sanity checks, so that is low priority.
function exact_trustregion(init::Vector, loc_s::AbstractVector, dat_s::AbstractVector,
                           d2funs::Vector{Vector{Function}}, opts::HODLR.Maxlikopts; profile::Bool=false,
                           vrb::Bool=false, dmax::Float64=1.0, dini::Float64=0.5,
                           eta::Float64=0.125, rtol::Float64=1.0e-8, atol::Float64=1.0e-5,
                           maxit::Int64=200, dcut::Float64=1.0e-4)
  dl, r1, st, cnt, fg = dini, 0.0, 0, 0, false
  xv = deepcopy(init)
  ro = zeros(length(init))
  gx = zeros(length(init))
  hx = zeros(length(init), length(init))
  for ct in 0:maxit
    cnt += 1
    vrb && println(xv)
    t1 = @elapsed begin
    if profile
      fx = exact_nlpl_objective(xv, gx, loc_s, dat_s, opts.kernfun, opts.dfuns, false)
    else
      fx = exact_nll_objective(xv, gx, loc_s, dat_s, opts.kernfun, opts.dfuns, false)
    end
    end
    vrb && println("objective+gradient  call took this long:  $(round(t1, digits=4))")
    t2 = @elapsed begin
    if profile
      hx .= exact_p_hessian(xv, loc_s, dat_s, opts.kernfun, opts.dfuns, d2funs)
    else
      hx .= exact_hessian(xv, loc_s, dat_s, opts.kernfun, opts.dfuns, d2funs)
    end
    end
    vrb && println("Hessian call took this long:              $(round(t2, digits=4))")
    vrb && println("Total time for the iteration:             $(round(t1+t2, digits=4))")
    vrb && println()
    # Solve the corresponding sub-problem:
    ro  .= _solve_subproblem_exact(gx, Symmetric(hx), dl)
    if profile
      fxp  = exact_nlpl_objective(xv.+ro, Array{Float64}(undef, 0), loc_s, dat_s, opts.kernfun,
                                  opts.dfuns, false)
    else
      fxp  = exact_nll_objective(xv.+ro, Array{Float64}(undef, 0), loc_s, dat_s, opts.kernfun,
                                 opts.dfuns, false)
    end
    r1   = _rho(fxp, xv, fx, gx, Symmetric(hx), ro)
    # Perform tests on solution of sub-problem:
    (r1 < 0.25)    && (dl = 0.25*norm(ro))
    (r1 > 0.75)    && (dl = min(2.0*norm(ro), dmax))
    (r1 > eta)     && (xv .+= ro)
    # Perform test for relative tolerance being reached:
    (fx != fxp && (isapprox(fx, fxp, rtol=rtol) || isapprox(fx, fxp, atol=atol))) && (fg = true)
    # Test for stopping the loop:
    fg             && (vrb && println("STOP: tolerance reached") ;          println() ; break)
    (norm(gx)<rtol)&& (vrb && println("STOP: Gradient reached tolerance") ; println() ; break)
    (dl<dcut)      && (vrb && println("STOP: Size of region too small")   ; println() ; break)
  end
  vrb && println("Total number of calls: $cnt")
  if profile
    pushfirst!(xv, exact_nlpl_scale(xv, loc_s, dat_s, opts.kernfun))
  end
  return cnt, xv
end

