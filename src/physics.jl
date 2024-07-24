#==============================================================================
# Functions specific to the problem's physics
# Currently solves Linear Advection.
==============================================================================#

struct PhysicsAndFluxParams
    dim::Int64
    numerical_flux_type::AbstractString
    pde_type::AbstractString
    include_source::Bool
    alpha_split::Float64
end

function calculate_numerical_flux(uM_face,uP_face,n_face,param::PhysicsAndFluxParams)

    #alpha = 0 #upwind
    #alpha = 1 #central
    #
    #f_numerical = 0.5 * a * (uM_face .+ uP_face) .+ a * (1-alpha) / 2.0 * (n_face) * (uM_face.-uP_face) # lin. adv, upwind/central
    #f_numerical = 0.25 * (uM_face.^2 .+ uP_face.^2) .+ (1-alpha) / 4.0 * (n_face) * (uM_face.^2 .-uP_face.^2) #Burgers, LxF
    #f_numerical = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) #Burgers, energy-conserving
    #if LxF
    #    stacked_MP = [uM_face;uP_face]
    #    #display(findmax(abs.([uM_face;uP_face]), dims=1))
    #    #lambda_ = 0.5 * stacked_MP[findmax(abs.(stacked_MP), dims=1)[2]]
    #    lambda_ = 0.5 * findmax(abs.(stacked_MP), dims=1)[1]
    #    #lambda_ = 0.5 * maximum(([uM_face;uP_face]), dims=1)
    #else
    #    lambda_ = 1/12.0 * (uM_face.-uP_face)
    #end

    #fluxM = 0.5 .* uM_face .*uM_face
    #fluxP = 0.5 .* uP_face .*uP_face
    #f_numerical = n_face .* 0.5*(fluxM .+ fluxP) #average
    #.+ lambda_ .* (uP_face .- uM_face)
    #display("f_numerical")
    #display(f_numerical)
    #return f_numerical

    f_numerical  = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) .* n_face # split
    if cmp(param.numerical_flux_type, "split_with_LxF")==0
        stacked_MP = [uM_face;uP_face]
        max_eigenvalue = findmax(abs.(stacked_MP))[1]
        f_numerical -= 0.5 .* max_eigenvalue .* (uP_face .- uM_face)
    end
    return f_numerical

end

function calculate_flux(u, Pi, param::PhysicsAndFluxParams)
    #f = a .* u # nodal flux for lin. adv.
    f = 0.5 .* (u.*u) # nodal flux

    #f_f = f[Fmask[:],:] #since we use GLL solution nodes, can select first and last element for face flux values.
    # what to do abt face? I don't use  Fmask anymore
    
    f_hat = Pi * f

    return f_hat#,f_f

end

function calculate_face_terms_nonconservative(chi_face, u_hat)
    return 0.5 * (chi_face * u_hat) .* (chi_face * u_hat)
end

function calculate_source_terms(x::AbstractVector{Float64},t::Float64, param::PhysicsAndFluxParams)
    #return zeros(size(x)) 
    if param.include_source
        return π*sin.(π*(x .- t)).*(1 .- cos.(π*(x .- t)))
    else
        return zeros(size(x))
    end
end
