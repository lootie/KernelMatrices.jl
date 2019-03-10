
using JLD

# Load in the data:
data    = load("../../data/precision_estimators.jld")

# Declare the precision function:
precfn  = (x,y)   -> norm(x-y)/max(norm(x), norm(y)) 
stinfng = (x,H,y) -> sqrt(dot(x-y, H\(x-y)))
stinfnh = (A,B)   -> sqrt(trace((A-B)*(inv(B)-inv(A))))

# Away from the MLE:
grad_rprec_far = map(x->precfn(x[1], x[2]),        zip(data["apx_grd_far"],
                                                       data["ext_grd_far"]))
hess_rprec_far = map(x->precfn(x[1], x[2]),        zip(data["apx_hes_far"],
                                                       data["ext_hes_far"]))


# At the MLE:
grad_rprec_mle = map(x->precfn(x[1], x[2]),        zip(data["apx_grd_mle"],
                                                       data["ext_grd_mle"]))
hess_rprec_mle = map(x->precfn(x[1], x[2]),        zip(data["apx_hes_mle"],
                                                       data["ext_hes_mle"]))
grad_sprec_mle = map(x->stinfng(x[1], x[2], x[3]), zip(data["apx_grd_mle"],
                                                       data["ext_fsh_mle"],
                                                       data["ext_grd_mle"]))
fish_sprec_mle = map(x->stinfnh(x[1], x[2]),       zip(data["apx_fsh_mle"],
                                                       data["ext_fsh_mle"]))


grad_r_far = log10.(mapslices(mean, grad_rprec_far, 1)[:])
hess_r_far = log10.(mapslices(mean, hess_rprec_far, 1)[:])
grad_r_mle = log10.(mapslices(mean, grad_rprec_mle, 1)[:])
hess_r_mle = log10.(mapslices(mean, hess_rprec_mle, 1)[:])
grad_s_mle = log10.(mapslices(mean, grad_sprec_mle, 1)[:])
fish_s_mle = log10.(mapslices(mean, fish_sprec_mle, 1)[:])

