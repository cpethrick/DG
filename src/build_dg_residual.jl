#==============================================================================
# Functions related to building the DG residual for an arbitrary problem.
==============================================================================#

include("set_up_dg.jl")
include("physics.jl")

function calculate_face_term(chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_f, alpha_split, u_hat, param::PhysicsAndFluxParams)

    f_numerical_dot_n = calculate_numerical_flux(uM_face,uP_face,n_face, a, param)
    face_flux_dot_n = f_f # For now, don't use face splitting and instead assume we always use collocated GLL nodes.
    face_flux_dot_n = alpha_split * f_f
    if alpha_split < 1
        face_flux_nonconservative = reshape(calculate_face_terms_nonconservative(chi_face, u_hat), size(f_f))
        face_flux_dot_n .+= (1-alpha_split) * face_flux_nonconservative
    end
    face_flux_dot_n .*= n_face
    
    face_term = chi_face' * W_f * (f_numerical_dot_n .- face_flux_dot_n)

    return face_term
end

function calculate_volume_terms(S_xi, f_hat)
    return S_xi * f_hat
end

function calculate_volume_terms_nonconservative(u, S_noncons, chi_v, u_hat) 
    return chi_v' * ((u) .* (S_noncons * u_hat))
end

function assemble_residual(u_hat, t, dg::DG, param::PhysicsAndFluxParams)
    rhs = zeros(Float64, size(u_hat))

    u = chi_v * u_hat # nodal solution
    f_hat,f_f = calculate_flux(u, Pi, Fmask)

    volume_terms = calculate_volume_terms(S_xi, f_hat)
    if alpha_split < 1
        volume_terms_nonconservative = calculate_volume_terms_nonconservative(u, S_noncons, chi_v, u_hat)
        volume_terms = alpha_split * volume_terms + (1-alpha_split) * volume_terms_nonconservative
    end

    face_terms = zeros(Float64, size(u_hat))
    uM = reshape(u[vmapM], (Nfaces,(size(u_hat))[2])) # size Nfaces * Nfp=1 x N_elem.
    uP = reshape(u[vmapP], (Nfaces,(size(u_hat))[2])) # size Nfaces * Nfp=1 x N_elem.
    for f in 1:Nfaces
        chi_face = chi_f[:,:,f]
        # hard code normal for now
        if f == 1
            n_face = -1
            #display("left face")
        else
            n_face = 1
            #display("right face")
        end
        uM_face = reshape(uM[f,:],(1,(size(u_hat))[2])) # size 1 x N_elem
        uP_face = reshape(uP[f,:],(1,(size(u_hat))[2]))
        f_face = reshape(f_f[f,:],(1,(size(u_hat))[2]))
        
        face_terms .+= calculate_face_term(chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_face, alpha_split, u_hat, param)
    end

    rhs = -1* M_inv * (volume_terms .+ face_terms)
    if param.include_source
        rhs+=calculate_source_terms(x,t)
    end
    return rhs
end
