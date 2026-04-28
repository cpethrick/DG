#==============================================================================
# Functions related to building the DG residual for an arbitrary problem.
==============================================================================#

include("set_up_dg.jl")
include("physics.jl")
include("parameters.jl")
include("cost_tracking.jl")

function hadamard_product(A::AbstractMatrix, B::AbstractMatrix, N_rows::Int, N_cols::Int)
    #returns C = A ⊙ B, element-wise multiplication of matrices A and B
    
    C = zeros(N_rows,N_cols)
    for irow = 1:N_rows
        for icol = 1:N_cols
            C[irow,icol] = A[irow,icol] * B[irow,icol]
        end
    end

    return C
    
end

function project(chi_project, u_hat, do_entropy_projection::Bool,le::LocalElement, dg::DG, param::PhysicsAndFluxParams)

    if do_entropy_projection
        return entropy_project(chi_project, u_hat,le, dg, param)
    else
        N_soln_input = round(Int,length(u_hat)/dg.N_state)
        N_nodes = size(chi_project)[1]
        u_projected = zeros(N_nodes*dg.N_state) # Take size from chi_project
        for istate = 1:dg.N_state
            u_projected[(istate-1)*N_nodes+1 : istate*N_nodes] = chi_project*u_hat[(istate-1)*N_soln_input+1 : istate*N_soln_input]
        end
        return u_projected
    end

end

function calculate_face_term(iface,istate, f_hat, u_hat, uM, uP, direction, le::LocalElement, dg::DG, param::PhysicsAndFluxParams)

    f_numerical = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], istate, direction,1,le, dg, param) #pass s.t. numerical flux chosen by problem physics.

    face_flux::AbstractVector{Float64} = le.phi_face[iface] * f_hat
    use_split::Bool = param.alpha_split < 1 && (direction == 1 || (direction == 2 && !param.usespacetime))
    if use_split
        face_flux*=param.alpha_split
        face_flux_nonconservative = calculate_face_terms_nonconservative(le.chi_face[iface], u_hat, direction,le, dg, param)
        face_flux .+= (1-param.alpha_split) * face_flux_nonconservative
    end

    #==
    display("iface" )
    display(iface)
    display("input face vals")
    display(uM)
    display(uP)
    display(dg.chi_face[iface]')
    display(dg.W_face[iface])
    display(dg.LFIDtoNormal[iface, direction])
    display(f_numerical)
    display(face_flux)
    ==#

    face_term = le.chi_face[iface]' * le.W_face[iface] * dg.LFIDtoNormal[iface, direction] * (f_numerical .- face_flux)

    return face_term
end

function calculate_face_numerical_flux_term(ielem, istate, iface, u_hat, uM, uP, direction, dg::DG, param::PhysicsAndFluxParams)
    # For the skew-symmetric stiffness operator, only the numerical flux part is needed.
    #


    le = dg.le[dg.EIDtoGroupID[ielem]]
    # EIDofexterior is used to detect the type of numerical flux to apply.
    EID_of_exterior = dg.EIDLFIDtoEIDofexterior[ielem,iface]
    f_numerical = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], istate, direction, EID_of_exterior, le, dg, param)
    face_term = le.chi_face[iface]' * le.W_face[iface] * dg.LFIDtoNormal[iface, direction] * (f_numerical)

    return face_term
end


function get_solution_at_face(find_interior_values::Bool, ielem, iface, u_hat_global, u_hat_local, dg::DG, param::PhysicsAndFluxParams)
    if dg.N_unique_GroupIDs > 1
        #display("More than 1 group ID will break get_solution_at_face()")
    end

    le = dg.le[dg.EIDtoGroupID[ielem]]
    if find_interior_values
        # interpolate to face
        u_face = project(le.chi_face[iface], u_hat_local, param.use_skew_symmetric_stiffness_operator, le, dg, param)
    else
        # Select the appropriate elem to find values from
        elem = dg.EIDLFIDtoEIDofexterior[ielem,iface]
        if elem > 0
            # Select the appropriate face of the neighboring elem
            face = dg.LFIDtoLFIDofexterior[iface]
            # find solution from the element whose ID was found 
            #Find local solution of the element of interest if we seek an exterior value
            u_hat_local_exterior_elem = zeros(le.N_soln_dof)
            for inode = 1:le.N_soln
                for istate = 1:dg.N_state
                    u_hat_local_exterior_elem[le.StIDLIDtoLSID[istate,inode]] = u_hat_global[dg.StIDGIDtoGSID[istate,dg.EIDLIDtoGID_basis[elem,inode]]]
                end
            end
            # interpolate to face
            u_face = project(le.chi_face[face], u_hat_local_exterior_elem, param.use_skew_symmetric_stiffness_operator, le,dg, param)
        elseif elem == 0 
            # elemID of 0 corresponds to Dirichlet boundary (weak imposition).
            # Find x and y coords of the face and pass to a physics function
            #for inode = 1:dg.N_soln_per_dim
                # The node numbering used in this code allows us
                # to choose the first N_soln_per_dim x-points
                # to get a vector of x-coords on the face.
             #   x_local[inode] = dg.x[dg.EIDLIDtoGID_vol[ielem,inode]]
                # The Dirichlet boundary doesn't depend on the y-coord, so leave it as zero.
                # y_local[inode] = dg.y[dg.EIDLIDtoGID_vol[ielem,inode]]
            #end
            if ielem > dg.N_elem_per_dim
                #assign u_face as interior value
                u_face = project(le.chi_face[iface], u_hat_local, param.use_skew_symmetric_stiffness_operator,le, dg, param)
            else
                #x_local = zeros(Float64, dg.N_face)
                y_local = zeros(Float64, le.N_face)
                x_local = dg.VX[ielem] .+ 0.5* (le.r_quad.+1) * dg.delta_x
                # The Dirichlet boundary doesn't depend on the y-coord, so leave it as zero.
                u_face = calculate_solution_on_Dirichlet_boundary(x_local, y_local, dg, param)
            end
        elseif elem == -1
            # elemID == -1 corresponds to outflow (transmissive) boundary
            # return interior value.
            # Note that this value shouldn't be used; returned as a dummy value.
            u_face =  zeros(le.N_face* dg.N_state)
        end

    end

    return u_face

end

function calculate_volume_terms(f_hat, direction, le::LocalElement, dg::DG)
    if direction == 1
        return le.S_xi * f_hat
    elseif direction==2
        return le.S_eta * f_hat
    end

end

function calculate_volume_terms_nonconservative(u, u_hat, direction, le::LocalElement, dg::DG, param::PhysicsAndFluxParams)
    if direction == 1 && occursin("burgers",param.pde_type) #both 1D and 2D burg
        # dimensions might be wrong here
        volume_term_physical = le.chi_quad' * ((u) .* (le.S_noncons_xi * u_hat))                                                                                                                              
    elseif direction == 2 && cmp(param.pde_type, "burgers2D") == 0
        volume_term_physical = le.chi_quad' * ((u) .* (le.S_noncons_eta * u_hat))
    else
        display("Warning: Nonconservative version only defined for Burgers!!")
        volume_term_physical = 0 * le.chi_quad' * ((u) .* (le.S_noncons_xi * u_hat)) #expensive way to get the right size
    end 
    return transform_physical_to_reference(volume_term_physical, direction, le, dg) 
end

function calculate_dim_cellwise_residual(ielem, istate, u_hat,u_hat_local,direction, dg::DG, param::PhysicsAndFluxParams)
    le = dg.le[dg.EIDtoGroupID[ielem]]

    rhs_local_state = zeros(Float64, le.N_soln)
    
    u_local_quadnodes = project(le.chi_quad, u_hat_local, false, le, dg, param) 

    f_hat_local_state = calculate_flux(u_local_quadnodes,direction,istate,le, dg, param)

    volume_terms_dim = calculate_volume_terms(f_hat_local_state,direction, le, dg)
    use_split::Bool = param.alpha_split < 1 && (direction== 1 || (direction== 2 && !param.usespacetime))
    if use_split
        volume_terms_nonconservative = calculate_volume_terms_nonconservative(u_local_quadnodes, u_hat_local,direction, le, dg, param)
        volume_terms_dim = param.alpha_split * volume_terms_dim + (1-param.alpha_split) * volume_terms_nonconservative
    end
    rhs_local_state += volume_terms_dim
    for iface in 1:dg.N_faces

        #How to get exterior values if those are all modal?? Would be doing double work...
        # I'm also doing extra work here by calculating the external solution one per dim. Think about how to improve this.
        uM = get_solution_at_face(true, ielem, iface, u_hat, u_hat_local, dg, param)
        uP = get_solution_at_face(false, ielem, iface, u_hat, u_hat_local, dg, param)

        rhs_local_state.+= calculate_face_term(iface,istate, f_hat_local_state, u_hat_local, uM, uP, direction, le, dg, param)

    end

    return rhs_local_state
end

function calculate_volume_terms_skew_symm(istate,u_hat_local, direction,le::LocalElement, dg::DG, param::PhysicsAndFluxParams)

    #u_local is only on volume nodes. Need to append u on face nodes.
    
    if dg.dim==1
        u_vf = zeros(le.N_quad+dg.N_faces*le.N_face, dg.N_state) # columns are states.
    else
        u_vf = zeros(le.N_quad+2*le.N_face+2*le.N_face_y, dg.N_state)
    end

    u_tilde_volume = entropy_project(le.chi_quad, u_hat_local,le, dg, param)
    for istate = 1:dg.N_state
        u_vf[1:le.N_quad, istate] = u_tilde_volume[(1:le.N_quad) .+ (istate-1) * le.N_quad]
    end
    face_ind_start=0
    for iface = 1:dg.N_faces
        u_tilde_face =  entropy_project(le.chi_face[iface], u_hat_local, le, dg, param)
        if dg.dim==1
            N_face=1
            display("Check that nothing got messsed dup for 1D")
        elseif iface < 3
            N_face = le.N_face_y
        else
            N_face = le.N_face
        end
        for istate = 1:dg.N_state
            u_vf[(1:N_face) .+ le.N_quad .+ face_ind_start, istate] = u_tilde_face[(1:N_face) .+ (istate-1) * N_face]
        end
        face_ind_start += N_face
    end

    
    #reference_two_point_flux = zeros(dg.N_quad+dg.N_faces*dg.N_face,dg.N_quad+dg.N_faces*dg.N_face)
    reference_two_point_flux = zeros(size(u_vf)[1],size(u_vf)[1])
    # Efficiency note: Some terms of QtildemQtildeT are zero, so those shouldn' be computed.
    # Can also take advantage of symmetry.
    for i =1:size(u_vf)[1]
        ui = u_vf[i,:]
        for j = 1:size(u_vf)[1]
            uj=u_vf[j,:]
            two_pt_flux = calculate_two_point_flux(ui, uj, direction,istate, le, dg, param)
            reference_two_point_flux[i,j] = two_pt_flux[1]
        end
    end

    volume_term = le.chi_vf * hadamard_product(le.QtildemQtildeT[:,:,direction], reference_two_point_flux, size(u_vf)[1], size(u_vf)[1]) * ones(size(u_vf)[1])

    return volume_term
end

function calculate_dim_cellwise_residual_skew_symm(ielem,istate,u_hat,u_hat_local,direction, dg::DG, param::PhysicsAndFluxParams)
    le = dg.le[dg.EIDtoGroupID[ielem]]
    rhs_local = zeros(Float64, le.N_soln)

    rhs_local .+= calculate_volume_terms_skew_symm(istate, u_hat_local,direction, le, dg, param)
    
    for iface in 1:dg.N_faces
        uM = get_solution_at_face(true, ielem, iface, u_hat, u_hat_local, dg, param)
        uP = get_solution_at_face(false, ielem, iface, u_hat, u_hat_local, dg, param)

        rhs_local .+= calculate_face_numerical_flux_term(ielem,istate, iface, u_hat_local, uM, uP, direction, dg, param)

    end
    
    return rhs_local
end

function assemble_local_state_residual(ielem,istate, u_hat, t, dg::DG, param::PhysicsAndFluxParams)
    le = dg.le[dg.EIDtoGroupID[ielem]]

    rhs_local_state = zeros(Float64, le.N_soln)
    u_hat_local = zeros(Float64, le.N_soln_dof)
    u_local = zeros(Float64, le.N_soln_dof)
    f_hat_local_state = zeros(Float64, le.N_soln)
    for inode = 1:le.N_soln
        for istate = 1:dg.N_state
            u_hat_local[le.StIDLIDtoLSID[istate,inode]] = u_hat[dg.StIDGIDtoGSID[istate,dg.EIDLIDtoGID_basis[ielem,inode]]]
        end
    end

    for idim = 1:dg.dim
        if param.use_skew_symmetric_stiffness_operator && idim==2 && param.strong_in_time
            # If param.strong_in_time is true, use strong DG formulation in the temporal dimension.
            rhs_local_state_dim = calculate_dim_cellwise_residual(ielem,istate,u_hat,u_hat_local,idim,dg,param)
        elseif param.use_skew_symmetric_stiffness_operator
            rhs_local_state_dim = calculate_dim_cellwise_residual_skew_symm(ielem,istate,u_hat,u_hat_local,idim,dg,param)
        else
            rhs_local_state_dim = calculate_dim_cellwise_residual(ielem,istate,u_hat,u_hat_local,idim,dg,param)
        end
        if idim == 2 && param.usespacetime
            rhs_local_state += -1 * le.M_inv * (rhs_local_state_dim)
        else
            rhs_local_state += -1 * le.MpK_inv * (rhs_local_state_dim)
        end
    end


    if param.include_source
        x_local = zeros(Float64, le.N_soln)
        y_local = zeros(Float64, le.N_soln)
        for inode = 1:le.N_soln
            x_local[inode] = dg.x[dg.EIDLIDtoGID_soln[ielem,inode]]
            y_local[inode] = dg.y[dg.EIDLIDtoGID_soln[ielem,inode]]
        end
        rhs_local_state+=le.Pi_soln*calculate_source_terms(istate,x_local,y_local,t, dg, param)
    end

    return rhs_local_state
end


function assemble_residual(u_hat, t, dg::DG, param::PhysicsAndFluxParams, subset_EIDs=nothing)
    # Parameter subset_EIDs assembles the residual ONLY in the subset of elements given by the vector subset_EIDs
    # All other residuals remain zero (i.e. will not change the solution on those elements)
    # If it's not passed, RHS is assembled in all cells.

    rhs = zeros(Float64, size(u_hat))

    if typeof(subset_EIDs) == Nothing
        elem_range = 1:dg.N_elem
    else
        elem_range = subset_EIDs
    end
    for ielem in elem_range

        for istate = 1:dg.N_state
            rhs_local = assemble_local_state_residual(ielem, istate, u_hat, t, dg, param)

            # store local rhs in global rhs
            for inode in 1:dg.le[dg.EIDtoGroupID[ielem]].N_soln
                rhs[dg.StIDGIDtoGSID[istate, dg.EIDLIDtoGID_basis[ielem,inode]]] = rhs_local[inode]
            end
        end
    end
    return rhs
end


function assemble_residual(u_hat, t, dg::DG, param::PhysicsAndFluxParams, ct::CostTracking, subset_EIDs=nothing) 

    update_CostTracking("assemble_residual", ct)

    return assemble_residual(u_hat, t, dg::DG, param::PhysicsAndFluxParams, subset_EIDs)

end
