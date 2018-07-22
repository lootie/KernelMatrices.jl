
using Rsvg, Plots, JLD
plotlyjs()

# Bring in the data:
data    = load("/path/to/materntraces.jld")
tr1_sym = data["tr1"]
tr2_sym = data["tr2"]
tr1_asm = data["as1"]
tr2_asm = data["as2"]
hutchN  = [1,2,3,4,5,10,20,30,50,100]
tru1    = data["tu1"]
tru2    = data["tu2"]
nsamp   = 50

##
#
# V1 (geoga-style)
#
##

# Make the plots:
pt1     = scatter(ones(nsamp).*hutchN[1]-0.1, tr1_asm[:,1], color=:blue, 
                  xlim=(0.0, hutchN[end]+1.0), xticks=(hutchN, hutchN),
                  size=(1000, 800), leg=false)
scatter!(ones(nsamp).*hutchN[1]+0.1, tr1_sym[:,1], color=:red, label="symmetric")
for j in 2:length(hutchN)
  scatter!(ones(nsamp).*hutchN[j]-0.1, tr1_asm[:,j], color=:blue, label="")
  scatter!(ones(nsamp).*hutchN[j]+0.1, tr1_sym[:,j], color=:red, label="")
end
plot!([tru1], linetype=:hline, color=:black)

pt2     = scatter(ones(nsamp).*hutchN[1]-0.1, tr1_asm[:,2], color=:blue, 
                  xlim=(0.0, hutchN[end]+1.0), xticks=(hutchN, hutchN),
                  size=(1000, 800), leg=false, ylim=(tru2-30.0, tru2+30.0))
scatter!(ones(nsamp).*hutchN[1]+0.1, tr2_sym[:,1], color=:red)
for j in 2:length(hutchN)
  scatter!(ones(nsamp).*hutchN[j]-0.1, tr2_asm[:,j], color=:blue, label="")
  scatter!(ones(nsamp).*hutchN[j]+0.1, tr2_sym[:,j], color=:red, label="")
end
plot!([tru2], linetype=:hline, color=:black)

##
#
# V2 (stein-style)
#
##

Plots.scalefontsizes(1.05)

std_tr1_sym = mapslices(std, tr1_sym .- tru1, 1)[:]
std_tr2_sym = mapslices(std, tr2_sym .- tru2, 1)[:]
std_tr1_asm = mapslices(std, tr1_asm .- tru1, 1)[:]
std_tr2_asm = mapslices(std, tr2_asm .- tru2, 1)[:]

pt21        = plot(hutchN, std_tr1_sym, color=:red, grid=false, leg=false, xaxis=:log10,
                   yaxis=:log10, xlim=(hutchN[1], hutchN[end]+15))
plot!(hutchN, std_tr1_asm, color=:blue)

pt22        = plot(hutchN, std_tr2_sym, color=:red, grid=false, leg=false, xaxis=:log10,
                   yaxis=:log10, xlim=(hutchN[1], hutchN[end]+15))
plot!(hutchN, std_tr2_asm, color=:blue)


