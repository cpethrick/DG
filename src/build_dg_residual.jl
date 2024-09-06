#==============================================================================
# Functions related to building the DG residual for an arbitrary problem.
==============================================================================#

include("set_up_dg.jl")
include("physics.jl")
include("parameters.jl")

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

function calculate_face_term(iface, f_hat, u_hat, uM, uP, direction, dg::DG, param::PhysicsAndFluxParams)
    f_numerical = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], direction,1,dg, param) #pass s.t. numerical flux chosen by problem physics.

    face_flux::AbstractVector{Float64} = dg.chi_f[:,:,iface] * f_hat
    use_split::Bool = param.alpha_split < 1 && (direction == 1 || (direction == 2 && !param.usespacetime))
    if use_split
        face_flux*=param.alpha_split
        face_flux_nonconservative = calculate_face_terms_nonconservative(dg.chi_f[:,:,iface], u_hat, direction, dg, param)
        face_flux .+= (1-param.alpha_split) * face_flux_nonconservative
    end

    face_term = dg.chi_f[:,:,iface]' * dg.W_f * dg.LFIDtoNormal[iface, direction] * (f_numerical .- face_flux)

    return face_term
end

function calculate_face_numerical_flux_term(ielem, iface, u_hat, uM, uP, direction, dg::DG, param::PhysicsAndFluxParams)
    # For the skew-symmetric stiffness operator, only the numerical flux part is needed.

    # EIDofexterior is used to detect the type of numerical flux to apply.
    EID_of_exterior = dg.EIDLFIDtoEIDofexterior[ielem,iface]
    f_numerical = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], direction,EID_of_exterior, dg, param)

    face_term = dg.chi_f[:,:,iface]' * dg.W_f * dg.LFIDtoNormal[iface, direction] * (f_numerical)

    return face_term
end


function get_solution_at_face(find_interior_values::Bool, ielem, iface, u_hat_global, u_hat_local, dg::DG, param::PhysicsAndFluxParams)

    if find_interior_values
        # interpolate to face
        u_face = dg.chi_f[:,:,iface]*u_hat_local
    else
        # Select the appropriate elem to find values from
        elem = dg.EIDLFIDtoEIDofexterior[ielem,iface]
        if elem > 0
            # Select the appropriate face of the neighboring elem
            face = dg.LFIDtoLFIDofexterior[iface]
            # find solution from the element whose ID was found 
            #Find local solution of the element of interest if we seek an exterior value
            u_hat_local_exterior_elem = zeros(dg.Np)
            for inode = 1:dg.Np
                u_hat_local_exterior_elem[inode] = u_hat_global[dg.EIDLIDtoGID_basis[elem,inode]]
            end
            # interpolate to face
            u_face = dg.chi_f[:,:,face] * u_hat_local_exterior_elem
        elseif elem == 0 
            # elemID of 0 corresponds to Dirichlet boundary (weak imposition).
            # Find x and y coords of the face and pass to a physics function
            x_local = zeros(Float64, dg.N_vol_per_dim)
            y_local = zeros(Float64, dg.N_vol_per_dim)
            for inode = 1:dg.N_vol_per_dim
                # The node numbering used in this code allows us
                # to choose the first N_vol_per_dim x-points
                # to get a vector of x-coords on the face.
                x_local[inode] = dg.x[dg.EIDLIDtoGID_vol[ielem,inode]]
                # The Dirichlet boundary doesn't depend on the y-coord, so leave it as zero.
                # y_local[inode] = dg.y[dg.EIDLIDtoGID_vol[ielem,inode]]
            end
            if ielem > dg.N_elem_per_dim
                #assign u_face as interior value
                u_face =  dg.chi_f[:,:,iface]*u_hat_local
            else
                u_face = calculate_solution_on_Dirichlet_boundary(x_local, y_local, param)
            end
        elseif elem == -1
            # elemID == -1 corresponds to outflow (transmissive) boundary
            # return interior value.
            # Note that this value shouldn't be used; returned as a dummy value.
            u_face =  dg.chi_f[:,:,iface]*u_hat_local
        end

    end

    return u_face

end

function calculate_volume_terms(f_hat, direction, dg)
    if direction == 1
        return dg.S_xi * f_hat
    elseif direction==2
        return dg.S_eta * f_hat
    end

end

function calculate_volume_terms_nonconservative(u, u_hat, direction, dg::DG, param::PhysicsAndFluxParams)
    if direction == 1 && occursin("burgers",param.pde_type) #both 1D and 2D burg
        volume_term_physical = dg.chi_v' * ((u) .* (dg.S_noncons_xi * u_hat))                                                                                                                              
    elseif direction == 2 && cmp(param.pde_type, "burgers2D") == 0
        volume_term_physical = dg.chi_v' * ((u) .* (dg.S_noncons_eta * u_hat))
    else
        volume_term_physical = 0 * dg.chi_v' * ((u) .* (dg.S_noncons_xi * u_hat)) #expensive way to get the right size
    end 
    return transform_physical_to_reference(volume_term_physical, direction, dg) 
end

function calculate_dim_cellwise_residual(ielem,u_hat,u_hat_local,u_local,direction, dg::DG, param::PhysicsAndFluxParams)

    rhs_local = zeros(Float64, dg.Np)
    
    f_hat_local = calculate_flux(u_local,direction, dg, param)

    volume_terms_dim = calculate_volume_terms(f_hat_local,direction, dg)
    use_split::Bool = param.alpha_split < 1 && (direction== 1 || (direction== 2 && !param.usespacetime))
    if use_split
        volume_terms_nonconservative = calculate_volume_terms_nonconservative(u_local, u_hat_local,direction, dg, param)
        volume_terms_dim = param.alpha_split * volume_terms_dim + (1-param.alpha_split) * volume_terms_nonconservative
    end
    rhs_local += volume_terms_dim
    for iface in 1:dg.Nfaces

        #How to get exterior values if those are all modal?? Would be doing double work...
        # I'm also doing extra work here by calculating the external solution one per dim. Think about how to improve this.
        uM = get_solution_at_face(true, ielem, iface, u_hat, u_hat_local, dg, param)
        uP = get_solution_at_face(false, ielem, iface, u_hat, u_hat_local, dg, param)

        rhs_local.+= calculate_face_term(iface, f_hat_local, u_hat_local, uM, uP, direction, dg, param)

    end

    return rhs_local
end

function calculate_volume_terms_skew_symm(u_local, u_hat_local, direction, dg::DG, param::PhysicsAndFluxParams)

    volume_term = zeros(Float64, dg.Np)

    #u_local is only on volume nodes. Need to append u on face nodes.
    
    u_vf = zeros(dg.Np+dg.Nfaces*dg.Nfp)
    u_vf[1:dg.Np] .= u_local
    for iface = 1:dg.Nfaces
        u_face=dg.chi_f[:,:,iface]*u_hat_local
        u_vf[(dg.Np+(iface-1)*dg.Nfp)+1:(dg.Np+(iface)*dg.Nfp)].=u_face
    end

    # Problem: how to select u_face? Alex's paper seems contradictory of whether we eant to select Nf * Nfp or Nfp. Can't possibly select Nfp to calculate the two point flux?
    reference_two_point_flux = zeros(dg.Np+dg.Nfaces*dg.Nfp,dg.Np+dg.Nfaces*dg.Nfp)
    # Efficiency note: Some terms of QtildemQtildeT are zero, so those shouldn' be computed.
    # Can also take advantage of symmetry.
    for i =1:length(u_vf)
        ui = u_vf[i]
        for j = 1:length(u_vf)
            uj=u_vf[j]
            reference_two_point_flux[i,j] = calculate_two_point_flux(ui, uj, direction, dg, param)
        end
    end

    volume_term = dg.chi_vf * hadamard_product(dg.QtildemQtildeT[:,:,direction], reference_two_point_flux, length(u_vf), length(u_vf)) * ones(length(u_vf))

    return volume_term
end

function calculate_dim_cellwise_residual_skew_symm(ielem,u_hat,u_hat_local,u_local,direction, dg::DG, param::PhysicsAndFluxParams)
    rhs_local = zeros(Float64, dg.Np)


    rhs_local .+= calculate_volume_terms_skew_symm(u_local, u_hat_local,direction, dg, param)
    
    for iface in 1:dg.Nfaces
        uM = get_solution_at_face(true, ielem, iface, u_hat, u_hat_local, dg, param)
        uP = get_solution_at_face(false, ielem, iface, u_hat, u_hat_local, dg, param)

        rhs_local.+= calculate_face_numerical_flux_term(ielem, iface, u_hat_local, uM, uP, direction, dg, param)

    end
    return rhs_local
end

function assemble_local_residual(ielem, u_hat, t, dg::DG, param::PhysicsAndFluxParams)
    rhs_local = zeros(Float64, dg.Np)
    u_hat_local = zeros(Float64, dg.Np)
    u_local = zeros(Float64, dg.Np)
    f_hat_local = zeros(Float64, dg.Np)
    for inode = 1:dg.Np
        u_hat_local[inode] = u_hat[dg.EIDLIDtoGID_basis[ielem,inode]]
    end

    u_local = dg.chi_v * u_hat_local # nodal solution
    # volume_terms = zeros(Float64, size(u_hat_local))
    # face_terms = zeros(Float64, size(u_hat_local))
    for idim = 1:dg.dim
        if param.use_skew_symmetric_stiffness_operator
            rhs_local += calculate_dim_cellwise_residual_skew_symm(ielem,u_hat,u_hat_local,u_local,idim,dg,param)
        else
            rhs_local += calculate_dim_cellwise_residual(ielem,u_hat,u_hat_local,u_local,idim,dg,param)
        end
    end

    rhs_local = -1* dg.M_inv * (rhs_local)

    if param.include_source
        x_local = zeros(Float64, dg.N_vol)
        y_local = zeros(Float64, dg.N_vol)
        for inode = 1:dg.N_vol
            x_local[inode] = dg.x[dg.EIDLIDtoGID_vol[ielem,inode]]
            y_local[inode] = dg.y[dg.EIDLIDtoGID_vol[ielem,inode]]
        end
        rhs_local+=dg.Pi*calculate_source_terms(x_local,y_local,t, param)
    end

    return rhs_local
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

        rhs_local = assemble_local_residual(ielem, u_hat, t, dg, param)

        # store local rhs in global rhs
        for inode in 1:dg.Np 
            rhs[dg.EIDLIDtoGID_basis[ielem,inode]] = rhs_local[inode]
        end
    end
    return rhs
end
