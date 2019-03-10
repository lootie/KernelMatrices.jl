
using LinearAlgebra, JLD, Plots, LaTeXStrings
plotlyjs()

function ellipse_pts(n::Int64, center::Vector{Float64}, Fish::Matrix{Float64}, q::Float64=5.991)
  # Get points on the circle of radius q:
  pts = collect(LinRange(0.0, 2.0*pi, n))
  pxy = vcat(cos.(pts)', sin.(pts)').*sqrt(q)
  # Get the Cholesky factor of the matrix:
  R   = cholesky(Fish).U
  # Return the new points on the ellipse:
  return mapslices(x->x+center, R\pxy, dims=1)
end

function plot_ellipse_sequence(H_C::Vector{Vector{Float64}}, H_V::Vector{Matrix{Float64}},
                               X_C::Vector{Vector{Float64}}, X_V::Vector{Matrix{Float64}},
                               colors, labels, ylab, legb)
  # Create the base plot:
  hpts = ellipse_pts(256, H_C[1], H_V[1])
  xpts = ellipse_pts(256, X_C[1], X_V[1])
  pt   = plot(hpts[1,:], hpts[2,:], color=colors[1], label=labels[1], xlabel="θ₀", ylabel=ylab,
              leg=legb, size=(1250, 625))
  plot!(xpts[1,:], xpts[2,:], color=colors[1], linestyle=:dash, label="")
  # Add on the extra indices:
  for j in 2:length(H_C)
    hpts = ellipse_pts(256, H_C[j], H_V[j])
    xpts = ellipse_pts(256, X_C[j], X_V[j])
    plot!(hpts[1,:], hpts[2,:], color=colors[j], label=labels[j])
    plot!(xpts[1,:], xpts[2,:], color=colors[j], linestyle=:dash, label="")
  end
  # Add the MLEs in the picture:
  for (hc, xc, cl)  in zip(H_C, X_C, colors)
    scatter!([hc[1]], [hc[2]], color=cl, markershape=:circle, markerstrokealpha=0.0, label="")
    scatter!([xc[1]], [xc[2]], color=cl, markershape=:square, markerstrokealpha=0.0, label="")
  end
  return pt
end


##
#
# Script component:
#
##

Plots.scalefontsizes(1.15)

big_data = load("../../../data/estimates_matern_bigrange.jld")
sml_data = load("../../../data/estimates_matern_smallrange.jld")

# make a test plot:
pt_big  = plot_ellipse_sequence(big_data["hodlr_fit"][1:3,1], big_data["hodlr_hes"][1:3,1], 
                                big_data["exact_fit"][1:3,1], big_data["exact_hes"][1:3,1], 
                                (:red, :blue, :green), ("log₂n=11", "log₂n=12", "log₂n=13" ), "", true)

pt_sml  = plot_ellipse_sequence(sml_data["hodlr_fit"][1:3,1], sml_data["hodlr_hes"][1:3,1], 
                                sml_data["exact_fit"][1:3,1], sml_data["exact_hes"][1:3,1], 
                                (:red, :blue, :green), ("", "", ""), "θ₁", false)



savefig(plot(pt_sml, pt_big, layout=(1,2)), "../../../report/figures/estimates_ellipses.eps")
