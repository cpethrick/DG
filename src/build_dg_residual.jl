#==============================================================================
# Functions related to building the DG residual for an arbitrary problem.
==============================================================================#

include("set_up_dg.jl")
include("physics.jl")

function calculate_face_term(iface, f_hat, u_hat, uM, uP, direction, dg::DG, param::PhysicsAndFluxParams)
    #chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_f, alpha_split, u_hat, param::PhysicsAndFluxParams)
    #modify to be one face and one element
    #display("iface")
    #display(iface)
    #display("direction")
    #display(direction)
    f_numerical_dot_n = calculate_numerical_flux(uM,uP,dg.LFIDtoNormal[iface,:], direction, param)

    #display("f_hat")
    #display(f_hat)
    #face_flux_dot_n::AbstractVector{Float64} = [(dg.chi_v * f_hat)[dg.LFIDtoLID[iface]]]
    face_flux_dot_n::AbstractVector{Float64} = dg.chi_f[:,:,iface] * f_hat
    if param.alpha_split < 1
        face_flux_dot_n*=param.alpha_split
        #face_flux_nonconservative = reshape(calculate_face_terms_nonconservative(dg.chi_f[:,:,iface], u_hat), size(face_flux_dot_n))
        face_flux_nonconservative = calculate_face_terms_nonconservative(dg.chi_f[:,:,iface], u_hat)
        face_flux_dot_n .+= (1-param.alpha_split) * face_flux_nonconservative
    end
    face_flux_dot_n .*= dg.LFIDtoNormal[iface]
    #display("face_flux_dot_n")
    #display(face_flux_dot_n)
    #display("f_numerical_dot_n")
    #display(f_numerical_dot_n)
    
    face_term = dg.chi_f[:,:,iface]' * dg.W_f * (f_numerical_dot_n .- face_flux_dot_n)
    #display("face_term")
    #display(face_term)

    return face_term
end

function get_solution_at_face(find_interior_values::Bool, ielem, iface, u_hat_global, u_hat_local, dg::DG)
    #display("Function get_solution_at_face")
    
    # Select the appropriate face to find values from
    if find_interior_values
        elem = ielem
        face = iface
    else
        elem = dg.EIDLFIDtoEIDofexterior[ielem,iface]
        face =dg.LFIDtoLFIDofexterior[iface]
    end

    #Find local solution of the element of interest if we seek an exterior value
    #Here, we assume that solution nodes are GLL when we pick the face value from the solution.
    if find_interior_values
        #u_face = u_local[dg.LFIDtoLID[face, :]]
        #u_face = u_local[dg.LFIDtoLID[face, :]]
        u_face = dg.chi_f[:,:,face]*u_hat_local
        display("interior")
        display(ielem)
        display(iface)
        #display(dg.LFIDtoLID[face, :])
    else
        u_hat_local_exterior_elem = zeros(dg.Np)
        for inode = 1:dg.Np
            u_hat_local_exterior_elem[inode] = u_hat_global[dg.EIDLIDtoGID_basis[elem,inode]]
        end
        #u_local_exterior_elem = dg.chi_v * u_hat_local_exterior_elem # nodal solution
        #u_face = u_local_exterior_elem[dg.LFIDtoLID[face,:]]
        u_face = dg.chi_f[:,:,face] * u_hat_local_exterior_elem
        display("exterior")
        display(ielem)
        display(iface)
        display(elem)
        display(face)
        #display(dg.LFIDtoLID[face,:])
    end
    #display("end Function get_solution_at_face")
    display(u_face)
    return u_face

end

function calculate_volume_terms(f_hat, direction, dg)
    if direction == 1
               #display("s_xi")
       #display(dg.S_xi)
       #display("f_hat")
       #display(f_hat)
       #display("prod")
       #display(dg.S_xi * f_hat) 
        return dg.S_xi * f_hat
    elseif direction==2
        return dg.S_eta * f_hat
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
    for ielem in 1:dg.N_elem
       #display("ielem")
       #display(ielem)
        ## Extract local solution
        ## Make local rhs vector
        ## find local u and f hat

        for inode = 1:dg.Np
            u_hat_local[inode] = u_hat[dg.EIDLIDtoGID_basis[ielem,inode]]
        end

        u_local = dg.chi_v * u_hat_local # nodal solution
        volume_terms = zeros(Float64, size(u_hat_local))
        face_terms = zeros(Float64, size(u_hat_local))
        for idim = 1:dg.dim
           #display("idim")
           #display(idim)
            f_hat_local = calculate_flux(u_local,idim, dg, param)
            ## Flux needs to be higher-dim!!

            volume_terms_dim = calculate_volume_terms(f_hat_local,idim, dg)
           #display("volume_terms_dim")
           #display(volume_terms_dim)
            if param.alpha_split < 1
                volume_terms_nonconservative = calculate_volume_terms_nonconservative(u_local, u_hat_local, dg)
                volume_terms_dim = param.alpha_split * volume_terms_dim + (1-param.alpha_split) * volume_terms_nonconservative
            end
            volume_terms += volume_terms_dim
            for iface in 1:dg.Nfaces
                
               #display("iface")
               #display(iface)
                #How to get exterior values if those are all modal?? Would be doing double work...
                # I'm also doing extra work here by calculating the external solution one per dim. Think about how to improve this.
                uM = get_solution_at_face(true, ielem, iface, u_hat, u_hat_local, dg)
               #display("uM")
               #display(uM)
                uP = get_solution_at_face(false, ielem, iface, u_hat, u_hat_local, dg)

                face_terms .+= calculate_face_term(iface, f_hat_local, u_hat_local, uM, uP, idim, dg, param)

            end
        end


        rhs_local = -1* dg.M_inv * (volume_terms .+ face_terms)


        if param.include_source
            x_local = zeros(Float64, dg.N_vol)
            for inode = 1:dg.N_vol
                x_local[inode] = dg.x[dg.EIDLIDtoGID_vol[ielem,inode]]
            end
            rhs_local+=dg.Pi*calculate_source_terms(x_local,t, param)
        end

        # store local rhs in global rhs
        for inode in 1:dg.Np 
            rhs[dg.EIDLIDtoGID_basis[ielem,inode]] = rhs_local[inode]
        end
    end
    return rhs
end
