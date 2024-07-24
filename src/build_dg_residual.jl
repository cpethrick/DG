#==============================================================================
# Functions related to building the DG residual for an arbitrary problem.
==============================================================================#

include("set_up_dg.jl")
include("physics.jl")

function calculate_face_term(iface, f_hat, u_hat, uM, uP, dg::DG, param::PhysicsAndFluxParams)
        #chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_f, alpha_split, u_hat, param::PhysicsAndFluxParams)
    #modify to be one face and one element
    f_numerical_dot_n = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface], param)

    face_flux_dot_n::AbstractVector{Float64} = [(dg.chi_v * f_hat)[dg.LFIDtoLID[iface]]]
    if param.alpha_split < 1
        face_flux_dot_n*=param.alpha_split
        #face_flux_nonconservative = reshape(calculate_face_terms_nonconservative(dg.chi_f[:,:,iface], u_hat), size(face_flux_dot_n))
        face_flux_nonconservative = calculate_face_terms_nonconservative(dg.chi_f[:,:,iface], u_hat)
        face_flux_dot_n .+= (1-param.alpha_split) * face_flux_nonconservative
    end
    face_flux_dot_n .*= dg.LFIDtoNormal[iface]
    
    face_term = dg.chi_f[:,:,iface]' * dg.W_f * (f_numerical_dot_n .- face_flux_dot_n)

    return face_term
end

function get_solution_at_face(find_interior_values::Bool, ielem, iface, u_hat_global, u_local, dg::DG)
    
    # Select the appropriate face to find values from
    if find_interior_values
        elem = ielem
        face = iface
    else
        elem = dg.EIDLFIDtoEIDofexterior[ielem,iface]
        #assume that face 1 always interfaces with face 2.
        if iface == 1
            face = 2
        else
            face = 1
        end
    end

    #Find local solution of the element of interest if we seek an exterior value
    #Here, we assume that solution nodes are GLL when we pick the face value from the solution.
    if find_interior_values
        #u_face = u_local[dg.LFIDtoLID[face, :]]
        u_face = u_local[dg.LFIDtoLID[face]]
        #display("interior")
        #display(ielem)
        #display(iface)
        #display(dg.LFIDtoLID[face])
    else
        u_hat_local_exterior_elem = zeros(size(u_local))
        for inode = 1:dg.Np
            u_hat_local_exterior_elem[inode] = u_hat_global[dg.EIDLIDtoGID[elem,inode]]
        end
        u_local_exterior_elem = dg.chi_v * u_hat_local_exterior_elem # nodal solution
        u_face = u_local_exterior_elem[dg.LFIDtoLID[face]]
        #display("exterior")
        #display(ielem)
        #display(iface)
        #display(elem)
        #display(face)
        #display(dg.LFIDtoLID[face])
    end
    return u_face

end

function calculate_volume_terms(f_hat, dg)
    if dg.dim == 1
        return dg.S_xi * f_hat
    else
        #Should I use a third dimension to indicate the direction of f_hat?
        return dg.S_xi * f_hat_xi + dg.S_eta * f_hat_eta
    end
        
end

function calculate_volume_terms_nonconservative(u, u_hat, dg) 
    return dg.chi_v' * ((u) .* (dg.S_noncons_xi * u_hat))
end

function assemble_residual(u_hat, t, dg::DG, param::PhysicsAndFluxParams)
    rhs = zeros(Float64, size(u_hat))
    u_hat_local = zeros(Float64, dg.Np)
    u_local = zeros(Float64, dg.Np)
    f_hat_local = zeros(Float64, dg.Np)
    for ielem in 1:dg.N_elem_per_dim
        ## Extract local solution
        ## Make local rhs vector
        ## find local u and f hat

        for inode = 1:dg.Np
            u_hat_local[inode] = u_hat[dg.EIDLIDtoGID[ielem,inode]]
        end

        u_local = dg.chi_v * u_hat_local # nodal solution
        f_hat_local = calculate_flux(u_local, dg.Pi, param)
        ## Flux needs to be higher-dim!!

        volume_terms = calculate_volume_terms(f_hat_local, dg)
        if param.alpha_split < 1
            volume_terms_nonconservative = calculate_volume_terms_nonconservative(u_local, u_hat_local, dg)
            volume_terms = param.alpha_split * volume_terms + (1-param.alpha_split) * volume_terms_nonconservative
        end

        face_terms = zeros(Float64, size(u_hat_local))
        for iface in 1:dg.Nfaces
            
            #How to get exterior values if those are all modal?? Would be doing double work...
            uM = get_solution_at_face(true, ielem, iface, u_hat, u_local, dg)
            uP = get_solution_at_face(false, ielem, iface, u_hat, u_local, dg)

            face_terms .+= calculate_face_term(iface, f_hat_local, u_hat_local, uM, uP, dg, param)
        end

        rhs_local = -1* dg.M_inv * (volume_terms .+ face_terms)


        if param.include_source
            x_local = zeros(Float64, dg.Np)
            for inode = 1:dg.Np
                x_local[inode] = dg.x[dg.EIDLIDtoGID[ielem,inode]]
            end
            rhs_local+=calculate_source_terms(x_local,t, param)
        end

        # store local rhs in global rhs
        for inode in 1:dg.Np 
            rhs[dg.EIDLIDtoGID[ielem,inode]] = rhs_local[inode]
        end
    end
    return rhs
end
