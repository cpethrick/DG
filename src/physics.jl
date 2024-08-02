#==============================================================================
# Functions specific to the problem's physics
# Currently solves Linear Advection.
==============================================================================#

include("set_up_dg.jl")

struct PhysicsAndFluxParams
    dim::Int64
    numerical_flux_type::AbstractString
    pde_type::AbstractString
    include_source::Bool
    alpha_split::Float64
    finaltime::Float64
    volumenodes::String #"GLL" or "GL"
    basisnodes::String #"GLL" or "GL"
    debugmode::Bool
end

function transform_physical_to_reference(f_physical, direction, dg::DG)
    return dg.C_m[direction,direction] * f_physical 
end

function calculate_numerical_flux(uM_face,uP_face,n_face, direction,dg::DG, param::PhysicsAndFluxParams)
    f_numerical=zeros(size(uM_face))

    #alpha = 0 #upwind
    #alpha = 1 #central
    if cmp(param.pde_type, "linear_adv_1D")==0
        a=1
        alpha = 0 
        if direction ==1 
            f_numerical = 0.5 * a * (uM_face .+ uP_face) .+ a * (1-alpha) / 2.0 * (n_face[direction]) * (uM_face.-uP_face) # lin. adv, upwind/central
        end
    end
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

   #display("normal in the direction of interest")
   #display(n_face[direction])
    #display("uM_face")
    #display(uM_face)
    if cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        f_numerical  = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) # split
        if cmp(param.numerical_flux_type, "split_with_LxF")==0
            stacked_MP = [uM_face;uP_face]
            max_eigenvalue = findmax(abs.(stacked_MP))[1]
            f_numerical += n_face[direction] * 0.5 .* max_eigenvalue .* (uM_face .- uP_face)
        end
    end
#==
    if cmp(param.pde_type,"burgers1D")==0
        if direction == 1
            f_numerical  = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) # split
            if cmp(param.numerical_flux_type, "split_with_LxF")==0
                max_eigenvalue = findmax(abs.(stacked_MP))[1]
                f_numerical += n_face[direction] * 0.5 .* max_eigenvalue .* (uM_face .- uP_face) # I think normal needs to be incorporated here.
            end
        elseif direction == 2
            f_numerical .+= 0 #for 1D burgers, no numerical flux in the y direction,
        end
    end
    ==#
    if dg.dim == 2
        # in 1D, C_m = 1 so we don't need this step
        f_numerical = transform_physical_to_reference(f_numerical, direction, dg)
    end
    return f_numerical

end

function calculate_flux(u, direction, dg::DG, param::PhysicsAndFluxParams)
    f = zeros(dg.N_vol)
            
    if cmp(param.pde_type,"linear_adv_1D")==0
        if direction == 1
            a = 1 #placeholder - should modify to actually be linear advection if I need that.
            f .+= a .* u # nodal flux for lin. adv.
        end
    elseif cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        f += 0.5 .* (u.*u) # nodal flux
    end

    if dg.dim == 2
        # in 1D, C_m = 1 so we don't need this step
        f = transform_physical_to_reference(f, direction, dg)
    end
    f_hat = dg.Pi * f

    return f_hat#,f_f

end

function calculate_face_terms_nonconservative(chi_face, u_hat)
    return 0.5 * (chi_face * u_hat) .* (chi_face * u_hat)
end

function calculate_source_terms(x::AbstractVector{Float64},y::AbstractVector{Float64},t::Float64, param::PhysicsAndFluxParams)
    if param.include_source
        if cmp(param.pde_type, "linear_adv_1D") ==0
            display("Warning! You probably don't want the source for linear advection!")
        end
        if cmp(param.pde_type, "burgers1D")==0
            return π*sin.(π*(x .- t)).*(1 .- cos.(π*(x .- t)))
        elseif cmp(param.pde_type, "burgers2D")==0 
            return π*sin.(π*(x .+ y.- sqrt(2) * t)).*(1 .- cos.(π*(x .+ y .- sqrt(2) * t)))
        end
    else
        return zeros(size(x))
    end
end
