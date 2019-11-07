
function _subproblem(xv::Vector, fx::Float64, gx::Vector, hx::Symmetric{Float64, Matrix{Float64}}, pv::Vector)
  return fx + dot(gx, pv) + 0.5*dot(pv,hx*pv)
end

function _rho(objfxp::Float64, x::Vector, fx::Float64, gx::Vector, 
              hx::Symmetric{Float64, Matrix{Float64}}, p::Vector)
  numr = fx - objfxp
  denm = _subproblem(x,fx,gx,hx,zeros(length(p))) - _subproblem(x,fx,gx,hx,p)
  return numr/denm
end

function _solve_subproblem_exact(g::Vector, B::Symmetric{Float64, Matrix{Float64}}, del::Float64)
  lmin = max(0.0, -real(eigmin(B)))
  lval = lmin + 0.1
  pl   = zeros(length(g))
  for cnt in 0:10
    Bi    = Symmetric(B + I*lval)
    R     = cholesky(Bi).U
    pl    = -Bi\g
    ql    = R\pl
    lval += abs2(norm(pl)/norm(ql))*(norm(pl)-del)/del
    (lval < lmin + 1.0e-10) && break
  end
  return pl
end

function trustregion(init::Vector, loc_s::AbstractVector, dat_s::AbstractVector,
                     d2funs::Vector{Vector{Function}}, opts::Maxlikopts; profile::Bool=false,
                     vrb::Bool=false, dmax::Float64=1.0, dini::Float64=0.5, eta::Float64=0.125,
                     rtol::Float64=1.0e-8, atol::Float64=1.0e-5, maxit::Int64=200,
                     dcut::Float64=1.0e-4)
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
      fx = nlpl_objective(xv, gx, loc_s, dat_s, opts)
    else
      fx = nll_objective(xv, gx, loc_s, dat_s, opts)
    end
    end
    vrb && println("objective+gradient  call took this long:  $(round(t1, digits=4))")
    t2 = @elapsed begin
    if profile
      hx .= nlpl_hessian(xv, loc_s, dat_s, opts, d2funs)
    else
      hx .= nll_hessian(xv, loc_s, dat_s, opts, d2funs)
    end
    end
    vrb && println("Hessian call took this long:              $(round(t2, digits=4))")
    vrb && println("Total time for the iteration:             $(round(t1+t2, digits=4))")
    vrb && println()
    # Solve the corresponding sub-problem:
    ro  .= _solve_subproblem_exact(gx, Symmetric(hx), dl)
    if profile
      fxp  = nlpl_objective(xv.+ro, Array{Float64}(undef, 0), loc_s, dat_s, opts)
    else
      fxp  = nll_objective(xv.+ro, Array{Float64}(undef, 0), loc_s, dat_s, opts)
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
    pushfirst!(xv, nlpl_scale(xv, loc_s, dat_s, opts))
  end
  return cnt, xv
end

function grad_and_fisher_matrix(prm, loc_s, dat_s, opts)
  # initialize:
  g_out = zeros(length(prm))
  F_out = zeros(length(prm), length(prm))
  K     = KernelMatrix(loc_s, loc_s, prm, opts.kernfun)
  HK    = KernelHODLR(K, opts.epK, opts.mrnk, opts.lvl, nystrom=true, plel=opts.apll)
  HODLR.symmetricfactorize!(HK, plel=opts.fpll)
  # Loop over and fill in the arrays minus the diagonal corrections:
  for j in eachindex(prm)
    HKj      = DerivativeHODLR(K, opts.dfuns[j], HK, plel=opts.apll)
    trace_j  = mean(v->HODLR_trace_apply(HK, HKj, v), opts.saav)
    g_out[j] = 0.5*(trace_j - dot(dat_s, HK\(HKj*(HK\dat_s))))
    for k in j:length(prm)
      HKk    = DerivativeHODLR(K, opts.dfuns[k], HK, plel=opts.apll)
      if k == j
        F_out[j,j] = 0.5*mean(v->HODLR_hess_tr1_sym_diag(HK, HKj, v), opts.saav)
      else
        F_out[j,k] = 0.25*mean(v->HODLR_hess_tr1_sym_offdiag(HK, HKj, HKk, v), opts.saav)
      end
    end
  end
  # Loop again for the expected fisher and fill in the diagonal corrections:
  for j in eachindex(prm)
    for k in (j+1):length(prm)
      F_out[j,k] -= 0.5*(F_out[j,j] + F_out[k,k])
    end
  end
  return g_out, Symmetric(F_out)
end

function fisherscore(init, loc_s, dat_s, opts; vrb=false,
                     g_tol=1.0e-8, s_tol=1.0e-8, maxit=50)
  old, new = deepcopy(init), deepcopy(init)
  for j in 1:maxit
    gj, Fj = grad_and_fisher_matrix(old, loc_s, dat_s, opts)
    new    = old + Fj\gj
    abs2(norm(gj)) < g_tol && (vrb && println("STOP: gradient tol reached") ; break)
    abs2(norm(old-new)) < g_tol && (vrb && println("STOP: step tol reached") ; break)
  end
  return new
end

