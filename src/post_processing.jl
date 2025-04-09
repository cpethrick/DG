
import PyPlot
import Printf
import DelimitedFiles
import LinearAlgebra
import FastGaussQuadrature

include("set_up_dg.jl")
include("parameters.jl")
include("physics.jl")
include("build_dg_residual.jl")

function get_integrating_vector_for_conservation_checks(u_hat_local,dg::DG)
    # in the future, this will have functionality to choose entropy variables. For now, just a conservation chcek.
    ones_vector = ones(dg.N_soln)
    ones_hat = dg.Pi_soln  * ones_vector
    return u_hat_local
end

function calculate_numerical_flux_for_postprocessing(direction,iface,ielem,u_hat_local, u_hat_global, dg, param  )

    # get solution at faces
    uM = get_solution_at_face(true, ielem, iface, u_hat_global, u_hat_local, dg, param)
    uP = get_solution_at_face(false, ielem, iface, u_hat_global, u_hat_local, dg, param)

    f_numerical = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], 1, direction, 1, dg, param) #some parameters are hard-coded for linear advection!!
end

function calculate_conservation_spacetime(u_hat, dg::DG, param::PhysicsAndFluxParams)

    dim = dg.dim
    if dg.N_state > 1
        display("Warning: nstate>1 may break the conservation check!")
    end

    integration_at_surfaces = zeros(2,dg.N_elem_per_dim)
    volume_integration = zeros(3,dg.N_elem_per_dim)
    display("============================")

    integration_initial = 0.0
    integration_final = 0.0
    for ielem = 1:dg.N_elem
        if param.usespacetime
            itslab = dg.EIDtoTSID[ielem]
            u_hat_local = u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof]
            integrating_vector = get_integrating_vector_for_conservation_checks(u_hat_local,dg)

            if itslab == 1

                x_local = dg.VX[ielem] .+ 0.5* (dg.r_flux.+1) * dg.delta_x
                y_local = zeros(size(x_local)) # leave zero as this is the lower face
                u_face_bottom = calculate_solution_on_Dirichlet_boundary(x_local, y_local,dg, param)
            else
                u_face_bottom = project(dg.chi_face[:,:,3], u_hat_local, true, dg, param)
            end
            u_face_top = project(dg.chi_face[:,:,4], u_hat_local, true, dg, param)

            integration_at_surfaces[1,itslab]+= integrating_vector' * dg.chi_face[:,:,3]' *dg.J_face* dg.W_face * u_face_bottom
            integration_at_surfaces[2,itslab]+= integrating_vector' * dg.chi_face[:,:,4]' *dg.J_face* dg.W_face * u_face_top

            # TEMPORAL TERMS
            #

            display(" ")
            display(ielem)
            display(u_hat_local)
            
            f_hat_local = calculate_flux(dg.chi_soln*u_hat_local,2,1, dg, param)
            volume_integration[1,itslab] += u_hat_local' * dg.MpK * dg.D_tau * f_hat_local
            display(dg.MpK * dg.D_tau * f_hat_local)
            prod=dg.MpK * dg.D_tau * f_hat_local
            PyPlot.figure("Across 3 bottom surface")
            PyPlot.clf()
            PyPlot.plot(u_hat_local[1:4], label="u hat")
            PyPlot.plot(f_hat_local[1:4], label="f_hat")
            PyPlot.plot(prod[1:4], label="dg.MpK * dg.D_tau * f_hat_local")
            PyPlot.legend()

            volume_integration[2,itslab] += u_hat_local' * dg.MpK * dg.L_tau3 * (
                                                             calculate_numerical_flux_for_postprocessing(2,3,ielem,u_hat_local, u_hat, dg, param)
                                                             - dg.phi_face[:,:,3] * f_hat_local )
            display(
                                                             calculate_numerical_flux_for_postprocessing(2,3,ielem,u_hat_local, u_hat, dg, param)
                                                             - dg.phi_face[:,:,3] * f_hat_local )
            volume_integration[2,itslab] += u_hat_local' * dg.L_tau4 * (
                                                             calculate_numerical_flux_for_postprocessing(2,4,ielem,u_hat_local, u_hat, dg, param)
                                                             - dg.phi_face[:,:,4] * f_hat_local )
            display(
                                                             calculate_numerical_flux_for_postprocessing(2,4,ielem,u_hat_local, u_hat, dg, param)
                                                             - dg.phi_face[:,:,4] * f_hat_local )
            if param.fluxreconstructionC > 0.0
                display("Error: the following assumes cdg!!")
                return NaN
            end
            #==
            f_hat_local = calculate_flux(dg.chi_soln*u_hat_local,2,1, dg, param)
            volume_integration[itslab] += -0.5 * u_hat_local' * dg.chi_face[:,:,3]' * dg.W_face * dg.LFIDtoNormal[3,2]  * (
                                                                                                                         dg.phi_face[:,:,3] * f_hat_local)
            volume_integration[itslab] += -0.5 * u_hat_local' * dg.chi_face[:,:,4]' * dg.W_face * dg.LFIDtoNormal[4,2]  * (
                                                                                                                         dg.phi_face[:,:,4] * f_hat_local)
#            if itslab == 1
                # add initial condition via numerical flux function
                volume_integration[itslab] += u_hat_local' * dg.chi_face[:,:,3]' * dg.W_face * dg.LFIDtoNormal[3,2]  * (
                                                        calculate_numerical_flux_for_postprocessing(2,3,ielem,u_hat_local, u_hat, dg, param) )
#            elseif itslab == dg.N_elem_per_dim
                # add upwind out of the tf surface
                volume_integration[itslab] += u_hat_local' * dg.chi_face[:,:,4]' * dg.W_face * dg.LFIDtoNormal[4,2]  * (
                                                        calculate_numerical_flux_for_postprocessing(2,4,ielem,u_hat_local, u_hat, dg, param) )
 #           end

==#
           
            # SPATIAL TERMS
            f_hat_local = calculate_flux(dg.chi_soln*u_hat_local,1,1, dg, param)
            volume_integration[3,itslab] +=  u_hat_local' * dg.S_xi * f_hat_local
            volume_integration[3,itslab] += u_hat_local' * dg.chi_face[:,:,1]' * dg.W_face * dg.LFIDtoNormal[1,1]  * (
                                                             calculate_numerical_flux_for_postprocessing(1,1,ielem,u_hat_local, u_hat, dg, param)
                                                             - dg.phi_face[:,:,1] * f_hat_local )
            volume_integration[3,itslab] += u_hat_local' * dg.chi_face[:,:,2]' * dg.W_face * dg.LFIDtoNormal[2,1]  * (
                                                             calculate_numerical_flux_for_postprocessing(1,2,ielem,u_hat_local, u_hat, dg, param)
                                                             - dg.phi_face[:,:,2] * f_hat_local )

        end
    end

    display("============================")

    if false
        fname = "conservation_check_"*string(dg.N_elem_per_dim)*"x"*string(dg.N_elem_per_dim)*"_"*param.fr_c_name*".csv"
        f = open(fname, "w")
        DelimitedFiles.writedlm(f, ["x" "JW_integration_bottom_face" "JW_integration_top_face" "volume_integration" "face_term_bottom_face" "face_term_top_face"], ",")
        DelimitedFiles.writedlm(f, [dg.VX[1:end] [integration_at_surfaces[1,:]' NaN]' [NaN integration_at_surfaces[2,:]']' [[NaN NaN NaN]; volume_integration']], ",")
        close(f)

    end

    display(integration_at_surfaces)

    #integration_initial = integration_at_surfaces[1,end]
    #integration_final = integration_at_surfaces[2,1]

    #display("Initial integrated state:")
    #display(integration_initial)
    #display("Final integrated state:")
    #display(integration_final)


    display("Volume integration")
    display(volume_integration')
    display("Sum over all timeslabs")
    display(sum(volume_integration,dims=2)')
    display("Sum of all terms in domain")
    display(sum(volume_integration))

    

    #conservation = integration_final - integration_initial
    #display("Conservation:")
    #display(conservation)
    return sum(volume_integration)
    #return conservation
end

function calculate_projection_error_local(u_face_interior, u_face_Dirichlet, ielem, dg::DG, param::PhysicsAndFluxParams)

    # get entropy potential and entropy variables at the face (interior soln)
    phi_face_interior = get_entropy_potential(u_face_interior, param)
    phi_face_Dirichlet = get_entropy_potential(u_face_Dirichlet, param)
    v_face_interior = get_entropy_variables(u_face_interior, param)
    v_face_Dirichlet = get_entropy_variables(u_face_Dirichlet, param)

    phi_jump = phi_face_interior - phi_face_Dirichlet
    v_jump = v_face_interior - v_face_Dirichlet

    vjumpTtimesu = zeros(size(phi_face_Dirichlet))
    for inode = 1:dg.N_vol_per_dim
        node_indices = dg.N_vol_per_dim*(1:dg.N_state) .- dg.N_vol_per_dim .+ inode
        vjumpTtimesu[inode] += (v_jump[node_indices])' * u_face_Dirichlet[node_indices]
    end

    return (phi_jump - vjumpTtimesu)' * dg.W_face * dg.J_face * ones(size(phi_jump))
end

function calculate_projection_corrected_entropy_change(u_hat, dg::DG, param::PhysicsAndFluxParams)
    projection_error = 0


    entropy_integration_at_surfaces = zeros(2,dg.N_elem_per_dim)
    entropy_initial = entropy_integration_at_surfaces[1,1]

    for ielem = 1:dg.N_elem
        itslab = dg.EIDtoTSID[ielem]
        u_hat_local = u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof]
        if itslab == 1
            # face on bottom (t=0)
            
            # get initial condition
            x_local = dg.VX[ielem] .+ 0.5* (dg.r_flux.+1) * dg.delta_x
            y_local = zeros(size(x_local)) # leave zero as this is the lower face
            u_face_Dirichlet = calculate_solution_on_Dirichlet_boundary(x_local, y_local, dg,param)
            u_face_interior = project(dg.chi_face[:,:,3],  u_hat_local,true,dg,param)

            s_vec = get_numerical_entropy_function(u_face_Dirichlet,param)
            entropy_integration_at_surfaces[1,1] += s_vec' * dg.W_face * dg.J_face * ones(size(s_vec))
            projection_error += calculate_projection_error_local(u_face_interior,u_face_Dirichlet, ielem, dg,param)#(phi_jump - vjumpTtimesu)' * dg.W_face * dg.J_face * ones(size(phi_jump))
        else
            # face on bottom of element
            u_face_bottom_interior = project(dg.chi_face[:,:,3],  u_hat_local,true,dg,param)
            s_vec = get_numerical_entropy_function(u_face_bottom_interior,param)
            entropy_integration_at_surfaces[1,itslab] += s_vec' * dg.W_face * dg.J_face * ones(size(s_vec))
        end

        u_face_top = project(dg.chi_face[:,:,4], u_hat_local, true, dg, param)
        s_vec = get_numerical_entropy_function(u_face_top,param)
        entropy_integration_at_surfaces[2,itslab] += s_vec' * dg.W_face * dg.J_face * ones(size(s_vec))
    
    end

    entropy_initial = entropy_integration_at_surfaces[1,1]
    entropy_final = entropy_integration_at_surfaces[2,end]

    display("Final entropy is ")
    display(entropy_final)
    display("Initial entropy from boundary condition is")
    display(entropy_initial)
    display("Projection error is ")
    display(projection_error)
    display("Entropy preservation:")
    display(entropy_final - entropy_initial + projection_error)
    display("Without projection correction:")
    display(entropy_final - entropy_initial)
    if true
        fname = "entropy_preservation_check.csv"
        f = open(fname, "w")
        DelimitedFiles.writedlm(f, [-1 projection_error 0.0], ",")
        DelimitedFiles.writedlm(f, [dg.VX[1:end] [entropy_integration_at_surfaces[1,:]' NaN]' [NaN entropy_integration_at_surfaces[2,:]']'], ",")
        close(f)

    end

    display(entropy_integration_at_surfaces)
    return entropy_final - entropy_initial + projection_error #return preservation

end


function display_plots(x_overint_1D,x_overint,y_overint,u0_overint_1D, u_calc_final_overint_1D, u_exact_overint_1D, u0_overint,u_calc_final_overint, u_exact_overint, dg, param)
    dim = dg.dim
    PyPlot.figure("Solution", figsize=(6,4))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks(dg.VX, minor=false)
    ax.xaxis.grid(true, which="major")
    PyPlot.plot(vec(x_overint_1D), vec(u0_overint_1D), label="initial")
    if param.include_source || cmp(param.pde_type, "linear_adv_1D")==0
        PyPlot.plot(vec(x_overint_1D), vec(u_exact_overint_1D), label="exact")
    end
    PyPlot.plot(vec(x_overint_1D), vec(u_calc_final_overint_1D), label="calculated")
    #Plots.plot!(vec(x_overint_1D), [vec(u_calc_final_overint_1D), vec(u0_overint_1D)], label=["calculated" "initial"])
    PyPlot.legend()
    pltname = string("plt", dg.N_elem_per_dim, ".pdf")
    PyPlot.savefig(pltname)

    PyPlot.figure("Grid", figsize=(6,6))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks(dg.VX, minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    if dim == 1
        PyPlot.axhline(0)
    elseif dim == 2
        ax.set_yticks(dg.VX, minor=false)
        ax.yaxis.grid(true, which="major", color="k")
    end
    PyPlot.plot(x_overint, y_overint,"o", color="yellowgreen", label="overintegration", markersize=0.25)
    PyPlot.plot(dg.x, dg.y, "o", color="darkolivegreen", label="volume nodes")
    pltname = string("grid", dg.N_elem_per_dim, ".pdf")
    PyPlot.savefig(pltname)

    for i in 1:length(u_calc_final_overint)
        if isnan(u_calc_final_overint[i])
            u_calc_final_overint[i]=0
        end
    end

    if dim == 2# && N_elem_per_dim == 4
        PyPlot.figure("Initial cond, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u0_overint, 20)
        PyPlot.colorbar()
        PyPlot.figure("Final soln, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u_calc_final_overint, 20)
        PyPlot.colorbar()
        PyPlot.figure("Final exact soln, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u_exact_overint, 20)
        PyPlot.colorbar()
    end
end

function post_process(u_hat, u_hat0, dg::DG, param::PhysicsAndFluxParams) 
    return post_process(u_hat, 0.0, u_hat0, dg::DG, param::PhysicsAndFluxParams)
end

function post_process(u_hat, current_time::Float64, u_hat0, dg::DG, param::PhysicsAndFluxParams)
    dim = dg.dim
    Np_overint_per_dim = dg.N_soln_per_dim+10
    Np_overint = (Np_overint_per_dim)^dim
    r_overint, w_overint = FastGaussQuadrature.gausslobatto(Np_overint_per_dim)
    (x_overint, y_overint) = build_coords_vectors(r_overint, dg)
    if dim==1
        chi_overint = vandermonde1D(r_overint,dg.r_basis)
        W_overint = LinearAlgebra.diagm(w_overint) # diagonal matrix holding quadrature weights
        J_overint = LinearAlgebra.diagm(ones(size(r_overint))*dg.J_soln[1]) #assume constant jacobian
    elseif dim==2
        chi_overint = vandermonde2D(r_overint, dg.r_basis, dg)
        W_overint = LinearAlgebra.diagm(vec(w_overint*w_overint'))
        J_overint = LinearAlgebra.diagm(ones(length(r_overint)^dim)*dg.J_soln[1]) #assume constant jacobian
    end

    u_exact_overint = calculate_exact_solution(x_overint, y_overint,Np_overint, current_time, dg, param) 
    u_calc_final_overint = zeros(size(x_overint))
    u0_overint = zeros(size(x_overint))
    u_calc_final = zeros(dg.N_vol*dg.N_elem)
    for ielem = 1:dg.N_elem
        #Extract only first state here
        u_hat_local = zeros(length(dg.r_basis)^dim) 
        u0_hat_local = zeros(size(u_hat_local)) 
        for inode = 1:dg.N_soln
            # only istate = 1, which is velocity for lin adv or burgers
            # and density for euler
            u_hat_local[inode] = u_hat[dg.StIDGIDtoGSID[1,dg.EIDLIDtoGID_basis[ielem,inode]]]
            u0_hat_local[inode] = u_hat0[dg.StIDGIDtoGSID[1,dg.EIDLIDtoGID_basis[ielem,inode]]]
        end
        u_calc_final_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u_hat_local
        u0_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u0_hat_local
        u_calc_final[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol] .= dg.chi_soln* u_hat_local
    end
    u_diff = u_calc_final_overint .- u_exact_overint

    x_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    u_calc_final_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    u0_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    u_exact_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    ctr = 1
    for iglobalID = 1:length(y_overint)
        if  y_overint[iglobalID] == 0.0
            x_overint_1D[ctr] = x_overint[iglobalID]
            u_calc_final_overint_1D[ctr] = u_calc_final_overint[iglobalID]
            u0_overint_1D[ctr] = u0_overint[iglobalID]
            u_exact_overint_1D[ctr] = u_exact_overint[iglobalID]
            ctr+=1
        end
    end
    if param.usespacetime
        # initial is 0.0, final is 2.0
        ctr0 = 1
        ctr2 = 1
        for iglobalID = 1:length(y_overint)
            if  y_overint[iglobalID] == 2.0
                u_calc_final_overint_1D[ctr2] = u_calc_final_overint[iglobalID]
                u_exact_overint_1D[ctr2] = u_exact_overint[iglobalID]
                ctr2+=1
            elseif y_overint[iglobalID] == 0.0
                u0_overint_1D[ctr0] = u_calc_final_overint[iglobalID]
                ctr0+=1
            end
        end
    end

    L2_error::Float64 = 0
    entropy_final_calc = 0
    entropy_initial = 0
    for ielem = 1:dg.N_elem
        L2_error += (u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint]') * W_overint * J_overint * (u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint])

        if !param.usespacetime
            entropy_final_calc += calculate_integrated_numerical_entropy(u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof], dg, param)
            entropy_initial += calculate_integrated_numerical_entropy(u_hat0[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof], dg, param)
        end
        
    end

    L2_error = sqrt(L2_error)

    Linf_error = maximum(abs.(u_diff))
    
    entropy_change = entropy_final_calc - entropy_initial

    conservation = 0.0
    if param.usespacetime
        proj_corrected_error = calculate_projection_corrected_entropy_change(u_hat, dg, param)
        entropy_change = proj_corrected_error

        if dg.N_state == 1
            conservation=calculate_conservation_spacetime(u_hat, dg, param)
        end
    else
        display("Warning: conservation check is not implemented for MoL!")
    end

   display_plots(x_overint_1D,x_overint, y_overint,u0_overint_1D, u_calc_final_overint_1D, u_exact_overint_1D, u0_overint,u_calc_final_overint,u_exact_overint, dg, param)

    return L2_error, Linf_error, entropy_change#, solution
end
