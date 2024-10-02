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
    f = (zeta.-1.0)./(zeta.+1.0)
    u = f .* f

    F = zeros(size(u))
    for i in 1:length(u)

        if u[i] < 1E-2
            F[i] = 1.0 .+ u[i]/3.0 .+ u[i].*u[i]/5.0 .+ u[i].*u[i].*u[i]/7.0
         else
             F[i] = 0.5 * log.(zeta[i]) ./ f[i]
         end
     end

     return (exterior_val .+ interior_val) ./ (2.0 * F)

end

function get_entropy_variables(solution, param::PhysicsAndFluxParams)

    if  occursin("burgers",param.pde_type)
        return solution
    elseif cmp(param.pde_type, "euler1D") == 0
        N_state = 3
        N_nodes = trunc(Int, length(solution)/N_state)
        entropy_variables = zeros(size(solution))

        gamm1 = 0.4
        gam = 1.4
        
        rho = solution[1:N_nodes]
        rhov = solution[N_nodes+1:N_nodes*2]
        E = solution[N_nodes*2+1:N_nodes*3]

        pressure = get_pressure_ideal_gas(solution)

        entropy = log.(pressure .* rho .^(-gam))

        rhoe = pressure/gamm1

        entropy_variables[1:N_nodes] = (rhoe .*(gam .+ 1.0 .- entropy) .- E)./rhoe
        entropy_variables[N_nodes+1:N_nodes*2] = rhov./rhoe
        entropy_variables[N_nodes*2+1:N_nodes*3] = -rho ./ rhoe

        return entropy_variables
    else
        display("Warning: entropy variables not defined for this PDE!")
        return solution
    end
end

function get_solution_variables(entropy_variables, param::PhysicsAndFluxParams)

    if  occursin("burgers",param.pde_type)
        return entropy_variables
    elseif cmp(param.pde_type, "euler1D") == 0
        N_state = 3
        N_nodes = trunc(Int, length(entropy_variables)/N_state)
        solution = zeros(size(entropy_variables))

        gamm1 =  0.4
        gam = 1.4
        
        e1 = entropy_variables[1:N_nodes]
        e2 = entropy_variables[N_nodes+1:N_nodes*2]
        e3 = entropy_variables[N_nodes*2+1:N_nodes*3]

        e_vel_sq = e2 .^2

        entropy = gam .- e1 .+ 0.5 * e_vel_sq ./ e3
        rhoe = (gamm1 ./ ((-1.0*e3).^gam)).^(1.0/gamm1) .* exp.(-1.0 * entropy / gamm1)

        solution[1:N_nodes] = -rhoe .* e3
        solution[N_nodes+1:N_nodes*2] = rhoe .* e2
        solution[N_nodes*2+1:N_nodes*3] = rhoe .* (1.0 .- 0.5 * e_vel_sq ./  e3)

        return solution
    else
        display("Warning: entropy variables not defined for this PDE!")
        return entropy_variables
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
    elseif cmp(param.pde_type, "euler1D")
        N_state = 3
        N_nodes = trunc(Int, length(solution)/N_state)
        entropy_variables = zeros(size(solution))

        gamm1 = 0.4
        gam = 1.4
        
        rho = solution[1:N_nodes]
        rhov = solution[N_nodes+1:N_nodes*2]
        E = solution[N_nodes*2+1:N_nodes*3]

        pressure = get_pressure_ideal_gas(solution)

        entropy = log.(pressure .* rho .^(-gam))

        return  -1.0 * rho .* entropy
    else
        display("Warning: entropy potential not defined for this PDE!")
        return solution
    end
end

function get_pressure_ideal_gas(solution)
    N_state = 3
    N_nodes = trunc(Int, length(solution)/N_state)

    gamm1 = 0.4


    rho = solution[1:N_nodes]
    rhov = solution[N_nodes+1:N_nodes*2]
    E = solution[N_nodes*2+1:N_nodes*3]
    pressure = (gamm1)* (E .- 0.5 * rhov.^2 ./rho)
    return pressure
end

function transform_physical_to_reference(f_physical, direction, dg::DG)
    return dg.C_m[direction,direction] * f_physical 
end

function entropy_project(chi_project, u_hat, dg::DG, param::PhysicsAndFluxParams)
    # chi_project is the basis functions evaluated at the desired projection points
    # (i.e., volume or face pts)
    # Per eq 42 in Alex Euler preprint
    u_volume_nodes = zeros(dg.N_dof)
    for istate = 1:dg.N_state
        u_volume_nodes[dg.StIDLIDtoLSID[istate, :]] = dg.chi_v * u_hat[dg.StIDLIDtoLSID[istate, :]]
    end
    v_volume = get_entropy_variables(u_volume_nodes, param)
    v_hat = zeros(dg.N_dof)
    for istate = 1:dg.N_state
        v_hat[dg.StIDLIDtoLSID[istate, :]] = dg.Pi * v_volume[dg.StIDLIDtoLSID[istate, :]]
    end

    N_nodes_proj = size(chi_project)[1]
    v_projected_nodes = zeros(N_nodes_proj* dg.N_state)
    for istate = 1:dg.N_state
        v_projected_nodes[(1:N_nodes_proj) .+ (dg.N_state-1)*N_nodes_proj] = chi_project * v_hat[dg.StIDLIDtoLSID[istate, :]]
    end

    return get_solution_variables(v_projected_nodes, param)

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
        elseif cmp(param.pde_type,"euler1D")==0 
            if param.usespacetime == true
                display("Warning!! Need to fix spacetime here!")
            end
             f_numerical = calculate_Ch_entropy_stable_flux(uM_face,uP_face,istate, dg, param)
        else
            display("Warning: No numerical flux is defined for this PDE type!!")
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


function calculate_Ch_entropy_stable_flux(uM,uP,istate, dg::DG, param::PhysicsAndFluxParams)
    N_nodes = length(uM) / dg.N_state # Calculating from uM to allow for generality betwen numerical flux and two-point flux.
    N_nodes = trunc(Int,N_nodes)

    f_Ch = zeros(N_nodes)
    u_node = zeros(dg.N_state)
    gam = 1.4
    gamm1 = 0.4
    for inode in 1:N_nodes
        node_indices = inode * 1:dg.N_state
        uM_node = uM[node_indices]
        uP_node = uP[node_indices]

        rho_ln = ln_average(uM_node[1], uP_node[1])
        pM = get_pressure_ideal_gas(uM_node)[1]
        pP = get_pressure_ideal_gas(uP_node)[1]

        betaM = uM_node[1]/(2.0*pM)
        betaP = uP_node[1]/(2.0*pP)
        beta_ln = ln_average(betaM,betaP)

        vM = uM_node[2]/uM_node[1]
        vP = uP_node[2]/uP_node[1]
        v_avg = 0.5*(vM+vP)
        v_square_avg = (v_avg)^2


        if istate == 1
            f_Ch[inode] = rho_ln * v_avg
        elseif istate == 2
            f_Ch[inode] = rho_ln * v_square_avg
        elseif istate == 3
            p_hat = 0.5 * (uM_node[1] + uP_node[1])/(betaM+betaP)

            enthalpy_hat = 1.0/(2.0 * beta_ln * gamm1) + v_square_avg + p_hat / rho_ln - 0.25 * (vM^2 + vP^2)
            f_Ch[inode] = rho_ln * v_avg * enthalpy_hat
        end

    end

    return f_Ch # physical flux

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
        flux_physical = calculate_Ch_entropy_stable_flux(ui, uj, istate, dg, param)
        #display("Warning!! Euler numerical flux has not been verified!")
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
        f = zeros(dg.Np) # calculated for one state at a time
        u_node = zeros(dg.N_state)
        for inode = 1:dg.Np
            u_node = u[dg.StIDLIDtoLSID[:,inode]]
            if istate == 1
                f_node = u_node[2]
            elseif istate == 2
                v = u_node[2]/u_node[1]
                p = get_pressure_ideal_gas(u)
                f_node = u_node[1] * v * v + p
            elseif istate == 3
                v = u_node[2]/u_node[1]
                p = get_pressure_ideal_gas(u_node)
                f_node = (u_node[3] +p)*v
            else
                display("Warning! There should only be three states for 1D Euler.")
                f_node = 0
            end
            f[inode] = f_node
        end
    elseif direction == 2  && cmp(param.pde_type, "euler1D") == 0
        # Time - physical flux will just be advective.
        f = u[dg.StIDLIDtoLSID[istate,inode]]
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

function calculate_initial_solution(dg::DG, param::PhysicsAndFluxParams)

    x = dg.x
    y = dg.y

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
    elseif cmp(param.pde_type, "euler1D") == 0
        u0 = calculate_euler_exact_solution(0, x, y, dg.Np, dg)
    else
        #u0 = 0.2* sin.(π * x) .+ 0.01
        u0 = sin.(π * (x))
    end
    return u0
end

function calculate_euler_exact_solution(t, x, y, Np, dg)

        # ordering is [elem1st1; elem1st2; elem1st3; elem2st1; elem2st2; ...]
        u_exact = zeros(dg.N_elem * Np*3) #Hard-code 3 states
        for ielem in 1:dg.N_elem
            node_indices = Np*(ielem-1) +1 : Np*ielem
            x_local = x[node_indices]
            y_local = y[node_indices]
            # Initial condition per 4.4 Friedrichs
            rho_local = 2 .+ sin.(2 * π * (x_local.-t))
            rhov_local = 2 .+ sin.(2 * π * (x_local.-t))
            E_local = (2 .+ sin.(2 * π * (x_local.-t))).^2
            # Indexing: 1:Np .+ Np*3*(ielem-1) .+ (istate-1)*Np
            u_exact[ (1:Np) .+ Np*3*(ielem-1) .+ (1-1)*Np] = rho_local
            u_exact[(1:Np) .+ Np*3*(ielem-1) .+ (2-1)*Np]= rhov_local
            u_exact[(1:Np) .+ Np*3*(ielem-1) .+ (3-1)*Np]= E_local

        end
        return u_exact
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

function calculate_source_terms(istate::Int, x::AbstractVector{Float64},y::AbstractVector{Float64},t::Float64, dg::DG, param::PhysicsAndFluxParams)
    if cmp(param.pde_type, "euler1D") != 0
        return calculate_source_terms(x, y, t, param)
    else
        if param.usespacetime
            display("Warning! Source will need to be checked when implementing space-time!")
        end
        # ordering is [elem1st1; elem1st2; elem1st3; elem2st1; elem2st2; ...]
        Q = zeros(dg.Np)
        gamm1=0.4
        for ielem in 1:dg.N_elem
            # Source per 4.5 Friedrichs
            if istate == 1
                Q .= 0
            elseif istate == 2 || istate == 3
                Q .= gamm1 * π * (7 .+ 4 * sin.( 2 * π * (x .- t))) .* cos.(2 * π * (x .- t))
            else
                display("Warning! There should only be 3 states.")
            end
        end
        return Q
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
