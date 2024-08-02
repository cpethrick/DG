#==============================================================================
# Functions related to building the DG residual for an arbitrary problem.
==============================================================================#

include("set_up_dg.jl")
include("physics.jl")

function calculate_face_term(iface, f_hat, u_hat, uM, uP, direction, dg::DG, param::PhysicsAndFluxParams)
    f_numerical = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], direction,dg, param)

    face_flux::AbstractVector{Float64} = dg.chi_f[:,:,iface] * f_hat
    if param.alpha_split < 1
        face_flux*=param.alpha_split
        face_flux_nonconservative = calculate_face_terms_nonconservative(dg.chi_f[:,:,iface], u_hat)
        face_flux .+= (1-param.alpha_split) * face_flux_nonconservative
    end
    
    face_term = dg.chi_f[:,:,iface]' * dg.W_f * dg.LFIDtoNormal[iface, direction] * (f_numerical .- face_flux)

    return face_term
end

function get_solution_at_face(find_interior_values::Bool, ielem, iface, u_hat_global, u_hat_local, dg::DG)
    
    # Select the appropriate face to find values from
    if find_interior_values
        elem = ielem
        face = iface
    else
        elem = dg.EIDLFIDtoEIDofexterior[ielem,iface]
        face =dg.LFIDtoLFIDofexterior[iface]
    end

    #Find local solution of the element of interest if we seek an exterior value
    if find_interior_values
        u_face = dg.chi_f[:,:,face]*u_hat_local
    else
        u_hat_local_exterior_elem = zeros(dg.Np)
        for inode = 1:dg.Np
            u_hat_local_exterior_elem[inode] = u_hat_global[dg.EIDLIDtoGID_basis[elem,inode]]
        end
        u_face = dg.chi_f[:,:,face] * u_hat_local_exterior_elem
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
        return dg.chi_v' * ((u) .* (dg.S_noncons_xi * u_hat))
    elseif direction == 2 && cmp(param.pde_type, "burgers2D") == 0 
        return dg.chi_v' * ((u) .* (dg.S_noncons_eta * u_hat))                                                                                                                                                     
    end
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
            if param.alpha_split < 1
                volume_terms_nonconservative = calculate_volume_terms_nonconservative(u_local, u_hat_local,idim, dg, param)
                volume_terms_dim = param.alpha_split * volume_terms_dim + (1-param.alpha_split) * volume_terms_nonconservative
            end
            volume_terms += volume_terms_dim
            for iface in 1:dg.Nfaces
                
                #How to get exterior values if those are all modal?? Would be doing double work...
                # I'm also doing extra work here by calculating the external solution one per dim. Think about how to improve this.
                uM = get_solution_at_face(true, ielem, iface, u_hat, u_hat_local, dg)
                uP = get_solution_at_face(false, ielem, iface, u_hat, u_hat_local, dg)

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
