
using Rsvg, Plots, JLD, LaTeXStrings
plotlyjs()

dat = load("/path/to/scaling_times.jld")
szs = log2.([2^10, 2^11, 2^12, 2^13, 2^14, 2^15, 2^16, 2^17, 2^18])
rks = [32, 72, 100]

xft1 = dat["exact_lik_times"]
hft1 = dat["hodlr_lik_times"]

xft2 = dat["exact_grd_times"]
hft2 = dat["hodlr_grd_times"]

xft3 = dat["exact_hes_times"]
hft3 = dat["hodlr_hes_times"]

function scatterslice!(pt::Plots.Plot, slice, sz, oset, color, msz)
  scatter!(ones(length(slice)).*(sz+oset), slice, color=color, markershape=:hexagon, markersize=msz,
           markerstrokealpha=0.0)
end

function plot_times(xft, hft, coef, ylab)
  mht1 = minimum(hft[1,:,:], 1)[:]
  mht2 = minimum(hft[2,:,:], 1)[:]
  mht3 = minimum(hft[3,:,:], 1)[:]
  if ylab
    ptt = plot(xticks=(szs, szs),xlim=(9.0, 19.0),yaxis=:log10,size=(500, 500),leg=false)
  else
    ptt = plot(xticks=(szs, szs),xlim=(9.0, 19.0),yaxis=:log10,size=(500, 500),leg=false)
  end
  for j in eachindex(szs)
    j <= 4 && (scatterslice!(ptt, [xft[j]], szs[j], 0.0, :blue, 4))
    scatterslice!(ptt, [minimum(hft[1,:,j])],  szs[j], 0.0, :orange, 4)
    scatterslice!(ptt, [minimum(hft[2,:,j])],  szs[j], 0.0, :red, 4)
    scatterslice!(ptt, [minimum(hft[3,:,j])],  szs[j], 0.0, :brown, 4)
  end
  plot!(szs[1:4], xft,  linestyle=:dot, color=:blue)
  plot!(szs,      mht1, linestyle=:dot, color=:orange)
  plot!(szs,      mht2, linestyle=:dot, color=:red)
  plot!(szs,      mht3, linestyle=:dot, color=:brown)
  plot!(szs, map(x->coef*x*log2(x)^2, map(y->2^y, szs)), color=:black, linestyle=:dash, linewidth=1.2)
  return ptt
end

Plots.scalefontsizes(1.15)
pts = map(x->plot_times(x[1], x[2], x[3], x[4]), ((xft1, hft1, 8.0e-7, true),
                                                  (xft2, hft2, 4.0e-6, false),
                                                  (xft3, hft3, 1.0e-5, false)))



