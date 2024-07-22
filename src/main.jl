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
include("build_dg_residual.jl")
include("set_up_dg.jl")

function setup_and_solve(N_elem,P,param::PhysicsAndFluxParams)
    # N_elem is number of elements
    # N is poly order
    
    # Limits of computational domain
    x_Llim = 0.0
    x_Rlim = 2.0

    dg = init_DG(P, 1, N_elem, [x_Llim,x_Rlim])

    #==============================================================================
    Start Up
    ==============================================================================#

#==
    # Number of local grid points
    Np = P+1

    # Number of faces
    Nfaces = 2
    
    NODETOL = 1E-10

    # Number of grid points on faces
    Nfp = 1
    
    # Array of points defining the extremes of each element
    VX = range(x_Llim,x_Rlim, N_elem+1) |> collect
    display(VX)

    # Vertex ID of left (1st col) and right (2nd col) extreme of element
    EtoV = collect(hcat((1:N_elem), 2:N_elem+1))
    
    # Index is global ID, values are local IDs
    GIDtoLID = mod.(0:(Np*N_elem.-1),Np).+1
    
    # Index of first dimension is element ID, index of second dimension is element ID
    # values are global ID
    EIDLIDtoGID = reshape(1:Np*N_elem, (Np,N_elem))' #note transpose

    # Index is local ID, value is local face ID
    # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    LIDtoLFID = zeros(Int64,Np)
    LIDtoLFID[[1,Np]] .= 1:Nfaces

    LFIDtoNormal = [-1,1] # normal of left face is 1, normal of right face is 1.
    
    # Index of first dim is element ID, index of second dimension is LID of an EDGE node.
    # 1D case: Nfp = 1.
    # values are global ID of the exterior node associated with the LID. Zero is not a face.
    # Possible future improvement - add some sort of numbering for the face nodes such that we store fewer zeros.
    EIDLIDtoGIDofexterior = zeros(Int64,N_elem,Np)
    EIDLIDtoGIDofexterior[:,1] = (0:N_elem-1)*Np
    EIDLIDtoGIDofexterior[:,Np] = (1:N_elem)*Np .+ 1  
    #manually assign periodic boundaries
    EIDLIDtoGIDofexterior[1,1]= N_elem*Np
    EIDLIDtoGIDofexterior[N_elem,Np] = 1
    #display(EIDLIDtoGIDofexterior)

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
    # All defined on a single element.
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
    #display(x)
    x_vector = ones(N_elem*Np)
    for  = 1:N_elem
        left_vertex_ID = EtoV[,1]
        right_vertex_ID = EtoV[,2]
        left_vertex_coord_phys = VX[left_vertex_ID]
        right_vertex_coord_phys = VX[right_vertex_ID]
        dx_local = right_vertex_coord_phys - left_vertex_coord_phys
        x_local = left_vertex_coord_phys .+ 0.5 * (r_volume .+ 1) * dx_local
        x_vector[EIDLIDtoGID[,1:Np]] .= x_local
    end
    display(x_vector) # matches old x.

    # Masks for edge nodes
    #fmask1 = findall( <(NODETOL), abs.(r_volume.+1))
    #fmask2 = findall( <(NODETOL), abs.(r_volume.-1))
    #Fmask = hcat(fmask1, fmask2)
    #Fx = x[Fmask[:],:]

    # Surface normals
    #nx = Normals1D(Nfp, Nfaces, N_elem)

    # Connectivity matrix
    #EtoE, EtoF = Connect1D(EtoV, Nfaces, N_elem)


    #vmapM,vmapP,vmapB,mapB,mapI,mapO,vmapI,vmapO = BuildMaps1D(EtoE, EtoF, N_elem, length(r_volume), Nfp, Nfaces, Fmask,x)
==#
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

    finaltime = 1.0

    u0 = sin.(π * dg.x) .+ 0.01
    #u0 = cos.(π * dg.x)
    #u_old = cos.(π * x)
    u_hat0 = zeros(N_elem*dg.Np)
    u_local = zeros(dg.Np)
    for ielem = 1:N_elem
        for inode = 1:dg.Np
            u_local[inode] = u0[dg.EIDLIDtoGID[ielem,inode]]
        end
        u_hat_local = dg.Pi*u_local
        #display(u_hat_local)
        u_hat0[dg.EIDLIDtoGID[ielem,:]] = u_hat_local
    end
    a = 2π


    #timestep size according to CFL
    #CFL = 0.01
    #xmin = minimum(abs.(x[1,:] .- x[2,:]))
    #dt = abs(CFL / a * xmin /2)
    dt = 1E-4
    Nsteps::Int64 = ceil(finaltime/dt)
    dt = finaltime/Nsteps

    u_hat = u_hat0
    current_time = 0
    residual = zeros(size(u_hat))
    rhs = zeros(size(u_hat))
    for tstep = 1:Nsteps
        for iRKstage = 1:nRKStage
            
            rktime = current_time + rk4c[iRKstage] * dt

            #####assemble residual
            rhs = assemble_residual(u_hat, rktime, dg, param)

            residual = rk4a[iRKstage] * residual .+ dt * rhs
            u_hat += rk4b[iRKstage] * residual
        end
        current_time += dt   
    end
    #==============================================================================
    Analysis
    ==============================================================================#

    Np_overint = dg.Np+10
    r_overint, w_overint = FastGaussQuadrature.gausslobatto(Np_overint)
    x_overint = ones(dg.N_elem_per_dim*Np_overint)
    for ielem = 1:dg.N_elem_per_dim
        x_local = dg.VX[ielem] .+ 0.5* (r_overint .+1) * dg.delta_x
        x_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= x_local
    end
    #x_overint = ones(length(r_overint)) * dg.VX[va]' + 0.5 * (r_overint .+ 1) * (dg.VX[vb]-dg.VX[va])'
    chi_overint = vandermonde1D(r_overint,dg.r_basis)
    W_overint = LinearAlgebra.diagm(w_overint) # diagonal matrix holding quadrature weights
    J_overint = LinearAlgebra.diagm(ones(size(r_overint))*dg.J[1]) #assume constant jacobian

    #u_exact_overint = sin.(2(x_overint .- a*finaltime))
    u_exact_overint = cos.(π*(x_overint.-current_time))
    u_calc_final_overint = zeros(size(x_overint))
    u0_overint = zeros(size(x_overint))
    u_calc_final = zeros(size(dg.x))
    for ielem = 1:dg.N_elem_per_dim
        u_hat_local = zeros(size(dg.r_volume)) 
        u0_hat_local = zeros(size(dg.r_volume)) 
        for inode = 1:dg.Np
            u_hat_local[inode] = u_hat[dg.EIDLIDtoGID[ielem,inode]]
            u0_hat_local[inode] = u_hat0[dg.EIDLIDtoGID[ielem,inode]]
        end
        u_calc_final_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u_hat_local
        u0_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u0_hat_local
        u_calc_final[(ielem-1)*dg.Np+1:(ielem)*dg.Np] .= dg.chi_v * u_hat_local
    end
    u_diff = u_calc_final_overint .- u_exact_overint

    ##################### Check later -- Why do I need to use diag?
    # I think it's because I'm doing all elements aggregate but should check..
    #display(u_diff)
    #display(W_overint)
    #display(J_overint)
    L2_error = 0
    energy_final_calc = 0
    energy_initial = 0
    for ielem = 1:dg.N_elem_per_dim
        L2_error += (sum((u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint]') * W_overint * J_overint * (u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint])))
        # use non-overintegrated qties to calculate energy difference
        energy_final_calc += sum(((u_calc_final[(ielem-1)*dg.Np+1:(ielem)*dg.Np]') * dg.W * dg.J * (u_calc_final[(ielem-1)*dg.Np+1:(ielem)*dg.Np])))
        energy_initial += sum(((u0[(ielem-1)*dg.Np+1:(ielem)*dg.Np]') * dg.W * dg.J * (u0[(ielem-1)*dg.Np+1:(ielem)*dg.Np])))
    end
    L2_error = sqrt(L2_error)

    energy_change = energy_final_calc - energy_initial

    Plots.vline(dg.VX, color="lightgray", linewidth=0.75, label="grid")
    Plots.plot!(vec(x_overint), [vec(u_exact_overint), vec(u_calc_final_overint)], label=["exact" "calculated"])
    pltname = string("plt", N_elem, ".pdf")
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
    P = 4 

    N_elem_range = [4 8 16 32 64 128 256]# 512 1024]
    #N_elem_range = [4]
    #N_elem_fine_grid = 1024 #fine grid for getting reference solution

    #_,_,reference_fine_grid_solution = setup_and_solve(N_elem_fine_grid,N)
    
    #alpha_split = 1 #Discretization of conservative form
    alpha_split = 2.0/3.0 #energy-stable split form
    
    param = PhysicsAndFluxParams("split", "burgers", false, alpha_split)

    L2_err_store = zeros(length(N_elem_range))
    energy_change_store = zeros(length(N_elem_range))

    for i=1:length(N_elem_range)

        # Number of elements
        N_elem =  N_elem_range[i]

        L2_err_store[i],energy_change_store[i] = setup_and_solve(N_elem,P,param)
    end

    Printf.@printf("P =  %d \n", P)

    Printf.@printf("n cells           Error     Error rate     Energy change \n")
    for i = 1:length(N_elem_range)
            conv_rate = 0.0
            if i>1
                conv_rate = log(L2_err_store[i]/L2_err_store[i-1]) / log(N_elem_range[i]/N_elem_range[i-1])
            end

            Printf.@printf("%d \t%.16f \t%.2f \t%.16f\n", N_elem_range[i], L2_err_store[i], conv_rate, energy_change_store[i])
    end
end

main()
