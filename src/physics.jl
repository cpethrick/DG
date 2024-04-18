#==============================================================================
# Functions specific to the problem's physics
# Currently solves Linear Advection.
==============================================================================#

function calculate_numerical_flux(uM_face,uP_face,n_face,a)

    alpha = 0 #upwind
    #alpha = 1 #central
    #
    f_numerical = 0.5 * a * (uM_face .+ uP_face) .+ a * (1-alpha) / 2.0 * (n_face) * (uM_face.-uP_face) 
end

function calculate_flux(u, Pi, a)
    f = a .* u # nodal flux

    f_f = f[[1,end], :] #since we use GLL solution nodes, can select first and last element for face flux values.
    
    f_hat = Pi * f

    return f_hat,f_f

end

