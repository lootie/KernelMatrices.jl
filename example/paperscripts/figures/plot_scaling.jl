
using Rsvg, Plots, JLD, LaTeXStrings
plotlyjs()

# Load the data files:
dat1 = load("../../../data/scaling_times_6workers.jld")

# Declare the sizes and ranks:
szs = log2.([2^10, 2^11, 2^12, 2^13, 2^14, 2^15, 2^16, 2^17, 2^18])
rks = [32, 72, 100]

# Declare easy-to-use variables for plotting functions:
xft1 = dat1["exact_lik_times"]
hft1 = dat1["hodlr_lik_times"]
xft2 = dat1["exact_grd_times"]
hft2 = dat1["hodlr_grd_times"]
xft3 = dat1["exact_hes_times"]
hft3 = dat1["hodlr_hes_times"]


function cubic_extrapolate(vals, locs, extra_locs)
  mat  = ones(length(locs), 2)
  mat[:,2] = map(x->(2^x)^3, locs)
  coef = (mat'mat)\(mat'vals)
  return map(x->coef[1] + coef[2]*(2^x)^3, extra_locs)
end

function scatterslice!(pt::Plots.Plot, slice, sz, oset, color, shape, msz)
  scatter!(ones(length(slice)).*(sz+oset), slice, color=color, markershape=shape, markersize=msz,
           markerstrokealpha=0.0)
end

function give_base_plot(ylab::Bool, xlims::Tuple{Float64, Float64}=(9.5, 18.5), 
                        sz::Tuple{Int64, Int64}=(500,500))
  if ylab
    ptt = plot(xticks=(szs,szs),xlim=xlims,yaxis=:log10,size=sz,leg=false,gridalpha=0.3)
  else
    ptt = plot(xticks=(szs,szs),xlim=xlims,yaxis=:log10,size=sz,leg=false,ylabel=nothing,gridalpha=0.3)
  end
  return ptt
end

function plot_times!(pt::Plots.Plot, xft, hft, coef, ticks, th_line)
  mht1 = minimum(hft[1,:,:], dims=1)[:]
  mht2 = minimum(hft[2,:,:], dims=1)[:]
  mht3 = minimum(hft[3,:,:], dims=1)[:]
  for j in eachindex(szs)
    j <= 4 && (scatterslice!(pt, [xft[j]], szs[j], 0.0, :blue, :cross, 4))
    scatterslice!(pt, [minimum(hft[1,:,j])],  szs[j], 0.0, :orange, ticks[1], 5)
    scatterslice!(pt, [minimum(hft[2,:,j])],  szs[j], 0.0, :red, ticks[2], 5)
    scatterslice!(pt, [minimum(hft[3,:,j])],  szs[j], 0.0, :brown, ticks[3], 5)
  end
  plot!(szs[1:4], xft,  linestyle=:dot, color=:blue)
  plot!(szs,      mht1, linestyle=:dot, color=:orange)
  plot!(szs,      mht2, linestyle=:dot, color=:red)
  plot!(szs,      mht3, linestyle=:dot, color=:brown)
  if th_line
    plot!(szs, map(x->coef*x*log2(x)^2, map(y->2^y, szs)), color=:black, linestyle=:dash, linewidth=1.2)
  end
  return pt
end

Plots.scalefontsizes(1.15)

pt1 = give_base_plot(true)
plot_times!(pt1, xft1, hft1, 8.0e-7, (:circle, :xcross, :utriangle), true)

pt2 = give_base_plot(true)
plot_times!(pt2, xft2, hft2, 4.0e-6, (:circle, :xcross, :utriangle), true)

pt3 = give_base_plot(true)
plot_times!(pt1, xft3, hft3, 1.3e-5, (:circle, :xcross, :utriangle), true)

Plots.savefig(pt1, "../../../report/figures/loglik_scaling.eps")
Plots.savefig(pt2, "../../../report/figures/gradient_scaling.eps")
Plots.savefig(pt3, "../../../report/figures/hessian_scaling.eps")

