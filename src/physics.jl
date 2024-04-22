#==============================================================================
# Functions specific to the problem's physics
# Currently solves Linear Advection.
==============================================================================#

function calculate_numerical_flux(uM_face,uP_face,n_face,a)

    alpha = 0 #upwind
    #alpha = 1 #central
    #
    #f_numerical = 0.5 * a * (uM_face .+ uP_face) .+ a * (1-alpha) / 2.0 * (n_face) * (uM_face.-uP_face) # lin. adv, upwind/central
    #f_numerical = 0.25 * (uM_face.^2 .+ uP_face.^2) .+ (1-alpha) / 4.0 * (n_face) * (uM_face.^2 .-uP_face.^2) #Burgers, LxF
    f_numerical = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) #Burgers, energy-conserving
    return f_numerical
end

function calculate_flux(u, Pi, a, Fmask)
    #f = a .* u # nodal flux for lin. adv.
    f = 0.5 .* (u.*u) # nodal flux

    f_f = f[Fmask[:],:] #since we use GLL solution nodes, can select first and last element for face flux values.
    
    f_hat = Pi * f

    return f_hat,f_f

end

