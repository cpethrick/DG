#==============================================================================
Problem definition

This code will solve the linear advection problem in 1D:
    ∂u/∂t + ∂f(u)/∂x = 0
    x ∈ [0, 2π]
    f(u) = 2π u (i.e., advection speed 2π)
    u(x,0) = uₒ(x) = sin(x) (initial condition)
    u(0,t)) = -sin(2π t) (inflow boundary condition)
==============================================================================#

#==============================================================================
Import packages
==============================================================================#

# For Jacobi/Legendre polynomials
# E.g., ClassicalOrthagonalPolynomials.jacobip(N, alpha, beta, x)
# NOTE: Need to normalize manually
import ClassicalOrthogonalPolynomials

#for gamma function
#import SpecialFunctions

# For Gauss quadrature points
# E.g., xi,wi = FastGaussQuadrature.gaussjacobi(N,0.0,0.0) is Gauss-Legendre nodes
#  Returns ([xi],[wi])
# E.g., FastGaussQuadrature.gausslobatto(N) is Gauss-Legendre-Lobatto nodes
import FastGaussQuadrature

import SparseArrays
import LinearAlgebra
import Printf
import Plots

#==============================================================================
Load external files
==============================================================================#

include("physics.jl")
include("FE_basis.jl")
include("FE_mapping.jl")

function calculate_face_term(chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_f, alpha_split, u_hat)


    f_numerical_dot_n = calculate_numerical_flux(uM_face,uP_face,n_face, a)
    face_flux_dot_n = f_f # For now, don't use face splitting and instead assume we always use collocated GLL nodes.
    face_flux_dot_n = alpha_split * f_f
    if alpha_split < 1
        face_flux_nonconservative = reshape(calculate_face_terms_nonconservative(chi_face, u_hat), size(f_f))
        face_flux_dot_n .+= (1-alpha_split) * face_flux_nonconservative
    end
    face_flux_dot_n .*= n_face
    
    face_term = chi_face' * W_f * (f_numerical_dot_n .- face_flux_dot_n)
    #display("face term")
    #display(face_term)

    return face_term
end

function calculate_volume_terms(S_xi, f_hat)

    return S_xi * f_hat
end

function calculate_volume_terms_nonconservative(u, S_noncons, chi_v, u_hat) 
    #step_1 = chi_v' * LinearAlgebra.diagm(u)
    #step_2 = step_1 * S_noncons
    #step_3 = step_2 * u_hat
    #display(u)
    #display((S_noncons * u_hat))
    return chi_v' * ((u) .* (S_noncons * u_hat))
end

function assemble_residual(u_hat, M_inv, S_xi, S_noncons, Nfaces, chi_f, W_f, Fmask, nx, a, Pi, chi_v, vmapM, vmapP, alpha_split, x, t)
    rhs = zeros(Float64, size(u_hat))

    u = chi_v * u_hat # nodal solution
    #display("u")
    #display(u)
    f_hat,f_f = calculate_flux(u, Pi, a, Fmask)

    volume_terms = calculate_volume_terms(S_xi, f_hat)
    #display("volume terms cons.")
    #display(volume_terms)
    if alpha_split < 1
        volume_terms_nonconservative = calculate_volume_terms_nonconservative(u, S_noncons, chi_v, u_hat)
        #display("volume terms noncons.")
        #display(volume_terms_nonconservative)
        volume_terms = alpha_split * volume_terms + (1-alpha_split) * volume_terms_nonconservative
        #display("volume terms w. avg.")
        #display(volume_terms)
    end

    face_terms = zeros(Float64, size(u_hat))
    uM = reshape(u[vmapM], (Nfaces,(size(u_hat))[2])) # size Nfaces * Nfp=1 x K.
    uP = reshape(u[vmapP], (Nfaces,(size(u_hat))[2])) # size Nfaces * Nfp=1 x K.
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
        uM_face = reshape(uM[f,:],(1,(size(u_hat))[2])) # size 1 x K
        #display("u-")
        #display(uM_face)
        uP_face = reshape(uP[f,:],(1,(size(u_hat))[2]))
        #display("u+")
        #display(uP_face)
        f_face = reshape(f_f[f,:],(1,(size(u_hat))[2]))
        #display("f_face")
        #display(f_face)
        
        face_terms .+= calculate_face_term(chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_face, alpha_split, u_hat)
    end

    source_terms = calculate_source_terms(x,t)

    rhs = -1* M_inv * (volume_terms .+ face_terms) + source_terms
    #display(rhs)
    return rhs
end

function setup_and_solve(K,N)
    # K is number of elements
    # N is poly order
    
    # Limits of computational domain
    x_Llim = 0.0
    x_Rlim = 2.0
    #display("dx")
    #display(x_Rlim / (K * (N+1)))

    # Array of points defining the extremes of each element
    VX = range(x_Llim,x_Rlim, K+1) |> collect

    #Coordinates of vertices of each element
    EtoV = collect(hcat((1:K), 2:K+1))

    #==============================================================================
    Start Up
    ==============================================================================#


    # Number of local grid points
    Np = N+1

    # Number of faces
    Nfaces = 2
    
    NODETOL = 1E-10

    # Number of grid points on faces
    Nfp = 1

    # Solution nodes - GLL
    # must choose GLL nodes unless I modify the selection of uP and uP for numerical flux.
    # Also will need to change splitting on the face.
    r_volume,w = FastGaussQuadrature.gausslobatto(Np)
    #r will be size N+1

    # Basis function nodes - GLL
    # Note: must choose GLL due to face flux splitting.
    #r_basis,w_basis=FastGaussQuadrature.gaussjacobi(Np,0.0,0.0)
    r_basis,w_basis=FastGaussQuadrature.gausslobatto(Np)

    #reference coordinates of L and R faces
    r_f_L::Float64 = -1
    r_f_R::Float64 = 1

    # Define Vandermonde matrices
    chi_v = vandermonde1D(r_volume,r_basis)
    d_chi_v_d_xi = gradvandermonde1D(r_volume,r_basis)
    chi_f = assembleFaceVandermonde1D(r_f_L,r_f_R,r_basis)
    
    W = LinearAlgebra.diagm(w) # diagonal matrix holding quadrature weights
    W_f = 1
    delta_x = VX[2]-VX[1]
    # constant jacobian on all elements as they are evenly spaced
    jacobian = delta_x/2 #reference element is 2 units long
    J = LinearAlgebra.diagm(ones(size(r_volume))*jacobian)

    # Mass and stiffness matrices as defined Eq. 9.5 Cicchino 2022
    M = chi_v' * W * J * chi_v
    M_inv = inv(M)
    S_xi = chi_v' * W * d_chi_v_d_xi
    S_noncons = W * d_chi_v_d_xi
    M_nojac = chi_v' * W * chi_v
    Pi = inv(M_nojac)*chi_v'*W

    # coords of nodes
    va = EtoV[:,1]
    vb = EtoV[:,2]
    x = ones(length(r_volume)) * VX[va]' + 0.5 * (r_volume .+ 1) * (VX[vb]-VX[va])'

    # Masks for edge nodes
    fmask1 = findall( <(NODETOL), abs.(r_volume.+1))
    fmask2 = findall( <(NODETOL), abs.(r_volume.-1))
    Fmask = hcat(fmask1, fmask2)
    Fx = x[Fmask[:],:]

    # Surface normals
    nx = Normals1D(Nfp, Nfaces, K)

    # Connectivity matrix
    EtoE, EtoF = Connect1D(EtoV, Nfaces, K)


    vmapM,vmapP,vmapB,mapB,mapI,mapO,vmapI,vmapO = BuildMaps1D(EtoE, EtoF, K, length(r_volume), Nfp, Nfaces, Fmask,x)

    #==============================================================================
    RK scheme
    ==============================================================================#

    if true 
        rk4a = [ 0.0,
            -567301805773.0/1357537059087.0,
            -2404267990393.0/2016746695238.0,
            -3550918686646.0/2091501179385.0,
            -1275806237668.0/842570457699.0];
        rk4b = [ 1432997174477.0/9575080441755.0,
            5161836677717.0/13612068292357.0,
            1720146321549.0/2090206949498.0,
            3134564353537.0/4481467310338.0,
            2277821191437.0/14882151754819.0];
        rk4c = [ 0.0,
            1432997174477.0/9575080441755.0,
            2526269341429.0/6820363962896.0,
            2006345519317.0/3224310063776.0,
            2802321613138.0/2924317926251.0];
        nRKStage=5
    end
    if false
        rk4a=[0]
        rk4b=[1]
        rk4c=[0]
        nRKStage=1
    end

    #==============================================================================
    ODE Solver 
    ==============================================================================#

    finaltime = 4.0

    u0 = sin.(π * x) .+ 0.01
    #u0 = cos.(π * x)
    #display(u0)
    u_hat0 = Pi * u0
    a = 2π

    #alpha_split = 1 #Discretization of conservative form
    alpha_split = 2.0/3.0 #energy-stable split form

    #timestep size according to CFL
    #CFL = 0.01
    #xmin = minimum(abs.(x[1,:] .- x[2,:]))
    #dt = abs(CFL / a * xmin /2)
    dt = 1E-4
    Nsteps::Int64 = ceil(finaltime/dt)
    dt = finaltime/Nsteps

    u_hat = u_hat0
    current_time = 0
    residual = zeros(Np, K)
    rhs = zeros(Np, K)
    for tstep = 1:Nsteps
        for iRKstage = 1:nRKStage
            rktime = current_time + rk4c[iRKstage] * dt

            #####assemble residual

            rhs = assemble_residual(u_hat, M_inv, S_xi, S_noncons, Nfaces, chi_f, W_f, Fmask, nx, a, Pi, chi_v, vmapM, vmapP, alpha_split,x,rktime)

            residual = rk4a[iRKstage] * residual .+ dt * rhs
            u_hat += rk4b[iRKstage] * residual
        end
        current_time += dt
        
    #    Np_overint = Np+10
    #    r_overint, w_overint = FastGaussQuadrature.gausslobatto(Np_overint)
    #    x_overint = ones(length(r_overint)) * VX[va]' + 0.5 * (r_overint .+ 1) * (VX[vb]-VX[va])'
    #    chi_overint = vandermonde1D(r_overint,r_basis)
    #    W_overint = LinearAlgebra.diagm(w_overint) # diagonal matrix holding quadrature weights
    #    J_overint = LinearAlgebra.diagm(ones(size(r_overint))*jacobian)

    #    #u_exact_overint = sin.(2(x_overint .- a*finaltime))
    #    u_calc_overint = chi_overint * u_hat
    #    Plots.vline(VX, color="lightgray", linewidth=0.75, label="grid")
    #    Plots.plot(vec(x_overint), [vec(u_calc_overint)], label=["calculated"])
    #    pltname = string("plt", K, ".pdf")
    #    Plots.savefig(pltname)
    #    sleep(0.02)
    end

    #==============================================================================
    Analysis
    ==============================================================================#

    Np_overint = Np+10
    r_overint, w_overint = FastGaussQuadrature.gausslobatto(Np_overint)
    x_overint = ones(length(r_overint)) * VX[va]' + 0.5 * (r_overint .+ 1) * (VX[vb]-VX[va])'
    chi_overint = vandermonde1D(r_overint,r_basis)
    W_overint = LinearAlgebra.diagm(w_overint) # diagonal matrix holding quadrature weights
    J_overint = LinearAlgebra.diagm(ones(size(r_overint))*jacobian)

    #u_exact_overint = sin.(2(x_overint .- a*finaltime))
    u_exact_overint = cos.(π*(x_overint.-finaltime))
    u_calc_final_overint = chi_overint * u_hat
    u_diff = u_calc_final_overint .- u_exact_overint

    ##################### Check later -- Why do I need to use diag?
    # I think it's because I'm doing all elements aggregate but should check..
    L2_error = sqrt(sum(LinearAlgebra.diag((u_diff') * W_overint * J_overint * (u_diff))))

    u0_overint = chi_overint * u_hat0

    # use non-overintegrated qties to calculate energy difference
    energy_initial = sum(LinearAlgebra.diag((u0') * W * J * (u0)))
    u_final = chi_v * u_hat
    energy_final_calc =  sum(LinearAlgebra.diag((u_final') * W * J * (u_final)))
    energy_change = energy_final_calc - energy_initial

    Plots.vline(VX, color="lightgray", linewidth=0.75, label="grid")
    Plots.plot!(vec(x_overint), [vec(u_exact_overint), vec(u_calc_final_overint)], label=["exact" "calculated"])
    pltname = string("plt", K, ".pdf")
    Plots.savefig(pltname)


    return L2_error, energy_change#, solution
end

#==============================================================================
Global setup

Definition of domain
Discretize into elements
==============================================================================#

#main program
function main()

    # Polynomial order
    N = 4 

    K_range = [4 8 16 32 64 128 256]# 512 1024]
    #K_range = [4]
    #K_fine_grid = 1024 #fine grid for getting reference solution

    #_,_,reference_fine_grid_solution = setup_and_solve(K_fine_grid,N)
    

    L2_err_store = zeros(length(K_range))
    energy_change_store = zeros(length(K_range))

    for i=1:length(K_range)

        # Number of elements
        K =  K_range[i]

        L2_err_store[i],energy_change_store[i] = setup_and_solve(K,N)
    end

    Printf.@printf("P =  %d \n", N)

    Printf.@printf("n cells           Error     Error rate     Energy change \n")
    for i = 1:length(K_range)
            conv_rate = 0.0
            if i>1
                conv_rate = log(L2_err_store[i]/L2_err_store[i-1]) / log(K_range[i]/K_range[i-1])
            end

            Printf.@printf("%d \t%.16f \t%.2f \t%.16f\n", K_range[i], L2_err_store[i], conv_rate, energy_change_store[i])
    end
end

main()
