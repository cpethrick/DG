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

function calculate_face_term(chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_f)


    f_numerical = calculate_numerical_flux(uM_face,uP_face,n_face, a)

    face_term = chi_face' * W_f * n_face * (f_numerical .- f_f)
    return face_term
end

function assemble_residual(u_hat, M_inv, S_xi, Nfaces, chi_f, W_f, nx, a, Pi, chi_v, vmapM, vmapP)
    rhs = zeros(Float64, size(u_hat))

    u = chi_v * u_hat # modal solution
    f_hat,f_f = calculate_flux(u, Pi, a)

    volume_terms = calculate_volume_terms(S_xi, f_hat)

    face_terms = zeros(Float64, size(u_hat))
    uM = reshape(u[vmapM], (Nfaces,(size(u_hat))[2])) # size Nfaces * Nfp=1 x K.
    uP = reshape(u[vmapP], (Nfaces,(size(u_hat))[2])) # size Nfaces * Nfp=1 x K.
    for f in 1:Nfaces
        chi_face = chi_f[:,:,f]
        # hard code normal for now
        if f == 1
            n_face = -1
        else
            n_face = 1
        end
        uM_face = reshape(uM[f,:],(1,(size(u_hat))[2])) # size 1 x K
        uP_face = reshape(uP[f,:],(1,(size(u_hat))[2]))
        f_face = reshape(f_f[f,:],(1,(size(u_hat))[2]))
        
        face_terms .+= calculate_face_term(chi_face, W_f, n_face, f_hat, uM_face, uP_face, a, f_face)
    end

    rhs = -1 * M_inv * ( volume_terms .+ face_terms)
    return rhs
end

function setup_and_solve(K,N)
    # K is number of elements
    # N is poly order
    
    # Limits of computational domain
    x_Llim = 0.0
    x_Rlim = 2.0*π

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
    r_volume,w = FastGaussQuadrature.gausslobatto(Np)
    #r will be size N+1

    # Integration nodes - GL
    r_basis,w_basis=FastGaussQuadrature.gaussjacobi(Np,0.0,0.0)

    #reference coordinates of L and R faces
    r_f_L::Float64 = -1
    r_f_R::Float64 = 1

    # Define Vandermonde matrices
    chi_v = vandermonde1D(r_volume,r_basis)
    #invV = inv(V)
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

    #==============================================================================
    ODE Solver 
    ==============================================================================#

    finaltime = π

    u0 = sin.(2x)
    #display(u0)
    u_hat0 = Pi * u0
    a = 2π

    alpha = 0 #upwind
    #alpha = 1 #central


    #timestep size according to CFL
    CFL = 0.75
    xmin = minimum(abs.(x[1,:] .- x[2,:]))
    dt = abs(CFL / a * xmin /2)
    Nsteps::Int64 = ceil(finaltime/dt)
    dt = finaltime/Nsteps

    u_hat = u_hat0
    current_time = 0
    residual = zeros(Np, K)
    rhs = zeros(Np, K)
    for tstep = 1:Nsteps
        for iRKstage = 1:5
            rktime = current_time + rk4c[iRKstage] * dt

            #####assemble residual

            rhs = assemble_residual(u_hat, M_inv, S_xi, Nfaces, chi_f, W_f, nx, a, Pi, chi_v, vmapM, vmapP)

            residual = rk4a[iRKstage] * residual .+ dt * rhs
            u_hat += rk4b[iRKstage] * residual
        end
        current_time += dt
    end

    #==============================================================================
    Analysis
    ==============================================================================#

    u_exact = vec(sin.(2(x .- a*finaltime)))
    u_calc_final = vec(chi_v * u_hat)

    L2_err = sqrt( sum( (u_calc_final .- u_exact).^2)/K)

    Plots.plot(vec(x), [u_calc_final,u_exact])
    pltname = string("plt", K, ".pdf")
    Plots.savefig(pltname)

    return L2_err
end

#==============================================================================
Global setup

Definition of domain
Discretize into elements
==============================================================================#

#main program
function main()

    # Polynomial order
    N = 3

    K_range = [4 8 16 32 64 128 256]
    L2_err_store = zeros(length(K_range))

    for i=1:length(K_range)


        # Number of elements
        K =  K_range[i]

        L2_err_store[i] = setup_and_solve(K,N)
    end

    Printf.@printf("N =  %d \n", N)

    for i = 1:length(K_range)
            conv_rate = 0.0
            if i>1
                conv_rate = log(L2_err_store[i]/L2_err_store[i-1]) / log(K_range[i]/K_range[i-1])
            end

            Printf.@printf("    %d    %.6f    %.2f\n", K_range[i], L2_err_store[i], conv_rate)
    end
end

main()
