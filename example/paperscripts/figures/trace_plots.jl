
using Statistics, Rsvg, Plots, JLD
plotlyjs()

# Bring in the data:
data    = load("../../../data/materntraces.jld")
tr1_sym = data["tr1"]
tr2_sym = data["tr2"]
tr1_asm = data["as1"]
tr2_asm = data["as2"]
hutchN  = [1,2,3,4,5,10,20,30,50,100]
tru1    = data["tu1"]
tru2    = data["tu2"]
nsamp   = 50

std_tr1_sym = mapslices(std, tr1_sym .- tru1, dims=1)[:]
std_tr2_sym = mapslices(std, tr2_sym .- tru2, dims=1)[:]
std_tr1_asm = mapslices(std, tr1_asm .- tru1, dims=1)[:]
std_tr2_asm = mapslices(std, tr2_asm .- tru2, dims=1)[:]

pt21        = plot(hutchN, std_tr1_sym, color=:red, grid=false, leg=false, xaxis=:log10,
                   yaxis=:log10, xlim=(hutchN[1], hutchN[end]+15), tickfont=font(12))
plot!(hutchN, std_tr1_asm, color=:blue, linestyle=:dash)

pt22        = plot(hutchN, std_tr2_sym, color=:red, grid=false, leg=false, xaxis=:log10,
                   yaxis=:log10, xlim=(hutchN[1], hutchN[end]+15), tickfont=font(12))
plot!(hutchN, std_tr2_asm, color=:blue, linestyle=:dash)

Plots.savefig(pt21, "../../../report/figures/trace_scale.eps")
Plots.savefig(pt22, "../../../report/figures/trace_range.eps")

