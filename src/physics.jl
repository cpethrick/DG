#==============================================================================
# Functions specific to the problem's physics
# Currently solves Linear Advection.
==============================================================================#

include("set_up_dg.jl")

struct PhysicsAndFluxParams
    dim::Int64
    numerical_flux_type::AbstractString
    pde_type::AbstractString
    usespacetime::Bool
    include_source::Bool
    alpha_split::Float64
    advection_speed::Float64
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
        a = param.advection_speed
        alpha = 0 
        if direction ==1 
            f_numerical = 0.5 * a * (uM_face .+ uP_face) .+ a * (1-alpha) / 2.0 * (n_face[direction]) * (uM_face.-uP_face) # lin. adv, upwind/central
        elseif direction == 2 && param.usespacetime 
            # second direction corresponding to time.
            # only use one-sided information such that the flow of information is from past to future.
            if n_face[direction] == -1
                # face is bottom. Use the information from the external element
                # which corresponds to the past
                f_numerical = uP_face
            elseif n_face[direction] == 1
                # face is bottom. Use internal solution
                # which corresonds to the past
                f_numerical = uM_face
            end
        end
    end
    
    if cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        f_numerical  = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) # split
        if cmp(param.numerical_flux_type, "split_with_LxF")==0
            stacked_MP = [uM_face;uP_face]
            max_eigenvalue = findmax(abs.(stacked_MP))[1]
            f_numerical += n_face[direction] * 0.5 .* max_eigenvalue .* (uM_face .- uP_face)
        end
    end
    
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
            f .+= param.advection_speed .* u # nodal flux for lin. adv.
        elseif direction == 2 && param.usespacetime
            f .+= u # CHECK
        end
    elseif cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        f += 0.5 .* (u.*u) # nodal flux
    end

    #display("f_physical")
    #display(f)
    if dg.dim == 2
        # in 1D, C_m = 1 so we don't need this step
        f = transform_physical_to_reference(f, direction, dg)
    #display(f)
    end
    f_hat = dg.Pi * f

    return f_hat#,f_f

end

function calculate_face_terms_nonconservative(chi_face, u_hat, direction, dg::DG, param::PhysicsAndFluxParams)
    if cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        u_physical = chi_face*u_hat
        f_physical =  0.5 * (u_physical) .* (u_physical)
        f_reference = transform_physical_to_reference(f_physical, direction, dg)
    else
        return 0*(chi_face*u_hat)
    end
    return f_reference
end 

function calculate_source_terms(x::AbstractVector{Float64},y::AbstractVector{Float64},t::Float64, param::PhysicsAndFluxParams)
    if param.include_source
        if cmp(param.pde_type, "linear_adv_1D") ==0
            display("Warning! You probably don't want the source for linear advection!")
        end
        if cmp(param.pde_type, "burgers1D")==0
            return π*sin.(π*(x .- t)).*(1 .- cos.(π*(x .- t)))
        elseif cmp(param.pde_type, "burgers2D")==0 
            display("Warning! This source is not correct!")
            return π*sin.(π*(x .+ y.- sqrt(2) * t)).*(1 .- cos.(π*(x .+ y .- sqrt(2) * t)))
        end
    else
        return zeros(size(x))
    end
end

function calculate_solution_on_Dirichlet_boundary(x::AbstractVector{Float64},y::AbstractVector{Float64})
    return sin.(π * (x)) .+ 0.01 # matches 1D linear advection initial condition
end
