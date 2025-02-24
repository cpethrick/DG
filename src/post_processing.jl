
function calculate_projection_corrected_entropy_change(u_hat, dg::DG, param::PhysicsAndFluxParams)
    entropy_initial = 0
    entropy_final = 0
    projection_error = 0

    for ielem = 1:dg.N_elem
        if ielem < dg.N_elem_per_dim+1
            # face on bottom (t=0)
            
            # get initial condition
            x_local = dg.VX[ielem] .+ 0.5* (dg.r_flux.+1) * dg.delta_x
            y_local = zeros(size(x_local)) # leave zero as this is the lower face
            u_face_Dirichlet = calculate_solution_on_Dirichlet_boundary(x_local, y_local, dg,param)
            
            u_hat_local = u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof]
            u_face_interior = project(dg.chi_face[:,:,3],  u_hat_local,true,dg,param)


            s_vec = get_numerical_entropy_function(u_face_Dirichlet,param)
            entropy_initial += s_vec' * dg.W_face * dg.J_face * ones(size(s_vec))


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


            projection_error += (phi_jump - vjumpTtimesu)' * dg.W_face * dg.J_face * ones(size(phi_jump))
        elseif ielem > dg.N_elem_per_dim^dg.dim - dg.N_elem_per_dim
            # face on top (t=tf)
            u_hat_local = u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof]
            u_face_interior = project(dg.chi_face[:,:,4],  u_hat_local,true,dg,param)

            s_vec = get_numerical_entropy_function(u_face_interior,param)
            entropy_final += s_vec' * dg.W_face * dg.J_face * ones(size(s_vec))
        end
    end

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
    return entropy_final - entropy_initial + projection_error #return preservation

end
