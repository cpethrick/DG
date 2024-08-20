#==============================================================================
# Functions related to building the DG residual for an arbitrary problem.
==============================================================================#

include("set_up_dg.jl")
include("physics.jl")
include("parameters.jl")

function calculate_face_term(iface, f_hat, u_hat, uM, uP, direction, dg::DG, param::PhysicsAndFluxParams)
    f_numerical = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], direction,dg, param)

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
            #display("Periodic boundary")
            #display("ielem")
            #display(ielem)
            #display("iface")
            #display(iface)
            #display("u_face")
            #display(u_face)
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

function assemble_residual(u_hat, t, dg::DG, param::PhysicsAndFluxParams)
    rhs = zeros(Float64, size(u_hat))
    u_hat_local = zeros(Float64, dg.Np)
    u_local = zeros(Float64, dg.Np)
    f_hat_local = zeros(Float64, dg.Np)
    for ielem in 1:dg.N_elem

        for inode = 1:dg.Np
            u_hat_local[inode] = u_hat[dg.EIDLIDtoGID_basis[ielem,inode]]
        end

        u_local = dg.chi_v * u_hat_local # nodal solution
        volume_terms = zeros(Float64, size(u_hat_local))
        face_terms = zeros(Float64, size(u_hat_local))
        for idim = 1:dg.dim
            f_hat_local = calculate_flux(u_local,idim, dg, param)

            volume_terms_dim = calculate_volume_terms(f_hat_local,idim, dg)
            use_split::Bool = param.alpha_split < 1 && (idim == 1 || (idim == 2 && !param.usespacetime))
            if use_split
                volume_terms_nonconservative = calculate_volume_terms_nonconservative(u_local, u_hat_local,idim, dg, param)
                volume_terms_dim = param.alpha_split * volume_terms_dim + (1-param.alpha_split) * volume_terms_nonconservative
            end
            volume_terms += volume_terms_dim
            for iface in 1:dg.Nfaces
                
                #How to get exterior values if those are all modal?? Would be doing double work...
                # I'm also doing extra work here by calculating the external solution one per dim. Think about how to improve this.
                uM = get_solution_at_face(true, ielem, iface, u_hat, u_hat_local, dg, param)
                uP = get_solution_at_face(false, ielem, iface, u_hat, u_hat_local, dg, param)

                face_terms .+= calculate_face_term(iface, f_hat_local, u_hat_local, uM, uP, idim, dg, param)

            end
        end

        rhs_local = -1* dg.M_inv * (volume_terms .+ face_terms)

        if param.include_source
            x_local = zeros(Float64, dg.N_vol)
            y_local = zeros(Float64, dg.N_vol)
            for inode = 1:dg.N_vol
                x_local[inode] = dg.x[dg.EIDLIDtoGID_vol[ielem,inode]]
                y_local[inode] = dg.y[dg.EIDLIDtoGID_vol[ielem,inode]]
            end
            rhs_local+=dg.Pi*calculate_source_terms(x_local,y_local,t, param)
        end

        # store local rhs in global rhs
        for inode in 1:dg.Np 
            rhs[dg.EIDLIDtoGID_basis[ielem,inode]] = rhs_local[inode]
        end
    end
    return rhs
end
