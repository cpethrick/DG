#==============================================================================
# Functions specific to the problem's physics
==============================================================================#

include("set_up_dg.jl")
include("parameters.jl")

function jump(exterior_val,interior_val)
    if size(interior_val) != size(exterior_val)
        display("Warning! Size mismatch in jump()!")
    end
    return exterior_val.-interior_val
end

function average(exterior_val, interior_val)
    if size(interior_val) != size(exterior_val)
        display("Warning! Size mismatch in average()!")
    end
    return 0.5 * (exterior_val.+interior_val)
end

function ln_average(exterior_val,interior_val)
    if size(interior_val) != size(exterior_val)
        display("Warning! Size mismatch in ln_average()!")
    end
    #Implementation per Appendix B [Ismail and Roe, 2009, Entropy-Consistent Euler Flux Functions II]
    zeta = exterior_val./interior_val
    f = (zeta.-1.0)/(zeta.+1.0)
    u = f .* f

    if u < 1E-2
         F = 1.0 .+ u/3.0 .+ u.*u/5.0 .+ u.*u.*u/7.0
     else
         F = 0.5 * log.(zeta) ./ f
     end

     return (exterior_val .+ interior_val) ./ (2.0 * F)

end

function get_entropy_variables(solution, param::PhysicsAndFluxParams)

    if  occursin("burgers",param.pde_type)
        return solution
    else
        display("Warning: entropy variables not defined for this PDE!")
        return solution
    end
end

function get_solution_variables(entropy_variables, param::PhysicsAndFluxParams)

    if  occursin("burgers",param.pde_type)
        return entropy_variables
    else
        display("Warning: entropy variables not defined for this PDE!")
        return solution
    end
end

function get_entropy_potential(solution, param::PhysicsAndFluxParams)
    if  occursin("burgers",param.pde_type)
        return 0.5 * solution .* solution 
    else
        display("Warning: entropy potential not defined for this PDE!")
        return solution
    end
end

function get_numerical_entropy_function(solution, param::PhysicsAndFluxParams)
    if  occursin("burgers",param.pde_type)
        return 0.5 * solution .* solution 
    else
        display("Warning: entropy potential not defined for this PDE!")
        return solution
    end
end

function get_pressure_ideal_gas(u)
    gamma = 1.4
    return (gamma-1)* (u[3] - 0.5 * u[2]^2/u[1])
end

function transform_physical_to_reference(f_physical, direction, dg::DG)
    return dg.C_m[direction,direction] * f_physical 
end

function entropy_project(chi_project, u_hat, dg::DG, param::PhysicsAndFluxParams)
    # chi_project is the basis functions evaluated at the desired projection points
    # (i.e., volume or face pts)
    v_volume = get_entropy_variables(dg.chi_v * u_hat, param)
    v_hat = dg.Pi * v_volume
    
    return get_solution_variables(chi_project * v_hat, param)

end


function calculate_numerical_flux(uM_face,uP_face,n_face, istate, direction, bc_type::Int, dg::DG, param::PhysicsAndFluxParams)
    # bc_type is int indicating the type of boundary
    # bc_type > 0 indicates boundary to another element (face within the domain
    #             or a periodic boundary condition), assigned according to 
    #             the problem physics
    # bc_type = 0 indicates a Dirichlet boundary for time dimension, returning the exterior value
    # bc_type = -1 indicates an outflow (transmissive) boundary for time dimension, returning the interior value 
    f_numerical=zeros(size(uM_face))

    if bc_type > 0
        # assign boundary according to problem physics.

        if direction == 2 && param.usespacetime 
            if param.spacetime_decouple_slabs == false
                # use decouple time slabs option as a proxy for applying U# as U*
                if cmp(param.pde_type, "burgers1D")
                    f_numerical = 0.5 * ( uM_face + uP_face )
                else
                    display("Warning: Need to impement returning of the two-pt flux")
                end
            else
                #apply upwind if decoupling time slabs.
                # second direction corresponding to time.
                # only use one-sided information such that the flow of information is from past to future.
                #
                # NOTE: must disable the decouple time slabs option if using a flux other than pure upwinding!
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
        elseif cmp(param.pde_type, "linear_adv_1D")==0
            a = param.advection_speed
            alpha = 0 #upwind
            #alpha = 1 #central
            if direction ==1 
                f_numerical = 0.5 * a * (uM_face .+ uP_face) .+ a * (1-alpha) / 2.0 * (n_face[direction]) * (uM_face.-uP_face) # lin. adv, upwind/central
            end # numerical flux only in x-direction.
        elseif cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
            f_numerical  = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) # split
            if cmp(param.numerical_flux_type, "split_with_LxF")==0
                stacked_MP = [uM_face;uP_face]
                max_eigenvalue = findmax(abs.(stacked_MP))[1]
                f_numerical += n_face[direction] * 0.5 .* max_eigenvalue .* (uM_face .- uP_face)
            end
        end
    elseif bc_type == 0
        # Dirichlet
            f_numerical = uP_face # uP_face has been set to be the analtical value in the 
                                  # get_solution_at_face() function 
    elseif bc_type == -1
        # Outflow
            f_numerical = uM_face # return interior solution
    else
        display("Warning: Numerical flux boundary type not recognized!")
    end

    if dg.dim == 2
        # in 1D, C_m = 1 so we don't need this step
        f_numerical = transform_physical_to_reference(f_numerical, direction, dg)
    end
    return f_numerical

end

function calculate_two_point_flux(ui,uj, direction, dg::DG, param::PhysicsAndFluxParams)

    if param.usespacetime && direction == 2
        flux_physical = 0.5 * (ui .+ uj)
    elseif cmp("burgers1D", param.pde_type) == 0 && direction == 2
        flux_physical = 0*ui
    elseif occursin("burgers", param.pde_type)
        flux_physical = 1.0/6.0 * (ui.* ui + ui .* uj + uj .* uj)
    else
        flux_physical = 0*ui
        display("Illegal PDE type! No two point flux is defined.")
    end
    
    return flux_physical
end

function calculate_two_point_flux(ui,uj, direction, istate::Int64, dg::DG, param::PhysicsAndFluxParams)
    f=0
    if cmp(param.pde_type, "euler1D") != 0
        # Redirect to scalar-valued version
        flux_physical = calculate_two_point_flux(ui,uj, direction, dg::DG, param::PhysicsAndFluxParams)
    elseif direction == 1  && cmp(param.pde_type, "euler1D") == 0
        # NOTE TO SELF: The input u to this function is a vector across the quad points in the cell.
        # Need to figure out indexing.
        #Add Ra flux here
        display("Warning!! Euler numerical flux has not been imlemented!")
    elseif direction == 2  && cmp(param.pde_type, "euler1D") == 0
        #From Eq. 3.5 of Friedrichs 2019
        if istate == 1
            flux_physical = ln_average(ui[1], uj[1])
        elseif istate == 2
            flux_physical = ln_average(ui[1], uj[1]) * average(ui[2]/ui[1], uj[2]/uj[1])
        elseif istate == 3
            pressurei = get_pressure_ideal_gas(ui)
            pressurej = get_pressure_ideal_gas(uj)
            betai = 0.5 * ui[1] / pressurei
            betaj = 0.5 * uj[1] / pressurej
            vi = ui[2]/ui[1]
            vj = ui[2]/ui[1]

            rho_ln = ln_average(ui[1],uj[1])
            beta_ln = ln_average(betai,betaj)

            gamma = 1.4

            flux_physical = 0.5 * rho_ln / (beta_ln * (gamma-1)) + rho_ln * (
                                        average(vi,vj)^2 - 0.5 * average(vi^2,vj^2)   )
        end
    end

    if length(f) > 1
        display("Warning! Flux is a vector, this will cause issues!!")
    end

    if dg.dim == 2
        return transform_physical_to_reference(flux_physical, direction, dg)
    else
        return flux_physical
    end
end


function calculate_flux(u, direction, dg::DG, param::PhysicsAndFluxParams)
    f = zeros(dg.N_vol)

    if direction == 2 && param.usespacetime
        f .+= u
    elseif cmp(param.pde_type,"linear_adv_1D")==0
        if direction == 1
            f .+= param.advection_speed .* u # nodal flux for lin. adv.
        end
    elseif cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        f += 0.5 .* (u.*u) # nodal flux
    end

    return f

end

function calculate_flux(u, direction, istate::Int64, dg::DG, param::PhysicsAndFluxParams)
    f=0
    if cmp(param.pde_type, "euler1D") != 0
        # Redirect to scalar-valued version
        f = calculate_flux(u, direction, dg::DG, param::PhysicsAndFluxParams)
    elseif direction == 1 && cmp(param.pde_type, "euler1D") == 0
        if istate == 1
            f = u[2]
        elseif istate == 2
            v = u[2]/u[1]
            p = get_pressure_ideal_gas(u)
            f = u[1] * v * v + p
        elseif istate == 3
            v = u[2]/u[1]
            p = get_pressure_ideal_gas(u)
            f = (u[3] +p)*v
        else
            display("Warning! There should only be three states for 1D Euler.")
            f = 0
        end
    elseif direction == 2  && cmp(param.pde_type, "euler1D") == 0
        # Time - physical flux will just be advective.
        f = u[istate]
    end

    # No pde_type check as the only vector-valued PDE is Euler
    # Reminder: conservative variables are u=[density, momentum, total energy]]
    #
    #
    # NOTE TO SELF: The input u to this function is a vector across the quad points in the cell.
    # Need to figure out indexing.

        
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
        display("Warning: Nonconservative is only defined for Burgers!")
        return 0*(chi_face*u_hat)
    end
    return f_reference
end 

function calculate_initial_solution(x::AbstractVector{Float64},y::AbstractVector{Float64}, param::PhysicsAndFluxParams)


    if param.usespacetime
        #u0 = cos.(π * (x))
        u0 = 0.1*sin.(π * (x))
        #u0 = 0*x
    elseif param.include_source && cmp(param.pde_type, "burgers2D")==0
        u0 = cos.(π * (x + y))
    elseif param.include_source && cmp(param.pde_type, "burgers1D")==0
        u0 = cos.(π * (x))
    elseif cmp(param.pde_type, "burgers2D") == 0
        u0 = exp.(-10*((x .-1).^2 .+(y .-1).^2))
    else
        #u0 = 0.2* sin.(π * x) .+ 0.01
        u0 = sin.(π * (x))
    end
    return u0
end

function calculate_source_terms(x::AbstractVector{Float64},y::AbstractVector{Float64},t::Float64, param::PhysicsAndFluxParams)
    if param.include_source
        if cmp(param.pde_type, "linear_adv_1D") ==0
            display("Warning! You probably don't want the source for linear advection!")
        end
        if cmp(param.pde_type, "burgers1D")==0 && param.usespacetime
            # y is time
            return π*sin.(π*(x .- y)).*(1 .- cos.(π*(x - y)))
        elseif cmp(param.pde_type, "burgers1D")==0
            return π*sin.(π*(x .- t)).*(1 .- cos.(π*(x .- t)))
        elseif cmp(param.pde_type, "burgers2D")==0 
            display("Warning! This source is not correct!")
            return π*sin.(π*(x .+ y.- sqrt(2) * t)).*(1 .- cos.(π*(x .+ y .- sqrt(2) * t)))
        end
    else
        return zeros(size(x))
    end
end

function calculate_source_terms(istate::Int, x::AbstractVector{Float64},y::AbstractVector{Float64},t::Float64, param::PhysicsAndFluxParams)
    if cmp(param.pde_type, "euler1D") != 0
        return calculate_source_terms(x, y, t, param)
    else
        display("Warning! Euler source terms not yet defined!!")
        return zeros(size(x))
    end
end

function calculate_solution_on_Dirichlet_boundary(x::AbstractVector{Float64},y::AbstractVector{Float64}, param::PhysicsAndFluxParams)

    if param.include_source
        return  cos.(π * (x-y))
    elseif cmp(param.pde_type, "burgers1D")==0
        #return 0.2*sin.(π * (x .- 0.314159265359878323))
        out = 0.2* ones(size(x))
        first1 = true # double-valued at 1.0 if using GLL, so force it to set 0 on left side
        for i in 1:length(x)
            xcoord = x[i]
            if xcoord == 1.0 && first1
                first1=false
            elseif xcoord >=  1.0
                out[i] = 0
            end
        end
        return out

    else
        return sin.(π * (x)) .+ 0.01
    end
end
