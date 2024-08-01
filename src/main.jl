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
import PyPlot

#==============================================================================
Load external files
==============================================================================#

include("physics.jl")
include("FE_basis.jl")
include("FE_mapping.jl")
include("build_dg_residual.jl")
include("set_up_dg.jl")

function setup_and_solve(N_elem_per_dim,P,param::PhysicsAndFluxParams)
    # N_elem_per_dim is number of elements PER DIM
    # N is poly order
    
    # Limits of computational domain
    x_Llim = 0.0
    x_Rlim = 2.0
    
    dim = param.dim 
    #==============================================================================
    Start Up
    ==============================================================================#
    dg = init_DG(P, dim, N_elem_per_dim, [x_Llim,x_Rlim], param.volumenodes, param.basisnodes)

    #==============================================================================
    RK scheme
    ==============================================================================#

    if param.debugmode == false
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
    else 
        rk4a=[0]
        rk4b=[1]
        rk4c=[0]
        nRKStage=1
    end

    #==============================================================================
    Initialization
    ==============================================================================#

    finaltime = param.finaltime

    if param.include_source
        u0 = cos.(π * dg.x)
    else
        u0 = sin.(π * (dg.x)) .+ 0.01
    end
    if dim == 2
    end
    #u_old = cos.(π * x)
    u_hat0 = zeros(dg.N_elem*dg.Np)
    u_local = zeros(dg.N_vol)
   #display(u0)
    for ielem = 1:dg.N_elem
        for inode = 1:dg.N_vol
            u_local[inode] = u0[dg.EIDLIDtoGID_vol[ielem,inode]]
        end
       #display(u_local)
        u_hat_local = dg.Pi*u_local
       #display(u_hat_local)
       #display(dg.chi_v*u_hat_local)#Check that transformation to/from modal is okay.
        u_hat0[dg.EIDLIDtoGID_basis[ielem,:]] = u_hat_local
    end
    a = 2π

    #timestep size according to CFL
    CFL = 0.01
    #xmin = minimum(abs.(x[1,:] .- x[2,:]))
    #dt = abs(CFL / a * xmin /2)
    dt = CFL * (dg.delta_x / dg.Np_per_dim)
    Nsteps::Int64 = ceil(finaltime/dt)
    dt = finaltime/Nsteps
    if param.debugmode == true
        Nsteps = 1
    end

    #==============================================================================
    ODE Solver 
    ==============================================================================#

    u_hat = u_hat0
    current_time = 0
    residual = zeros(size(u_hat))
    rhs = zeros(size(u_hat))
    display("dt")
    display(dt)
    for tstep = 1:Nsteps
    #for tstep = 1:1
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

    Np_overint_per_dim = dg.Np_per_dim+10
    Np_overint = (Np_overint_per_dim)^dim
    r_overint, w_overint = FastGaussQuadrature.gausslobatto(Np_overint_per_dim)
    (x_overint, y_overint) = build_coords_vectors(r_overint, dg)
    if dim==1
        chi_overint = vandermonde1D(r_overint,dg.r_basis)
        W_overint = LinearAlgebra.diagm(w_overint) # diagonal matrix holding quadrature weights
        J_overint = LinearAlgebra.diagm(ones(size(r_overint))*dg.J[1]) #assume constant jacobian
    elseif dim==2
        chi_overint = vandermonde2D(r_overint, dg.r_basis, dg)
        W_overint = LinearAlgebra.diagm(vec(w_overint*w_overint'))
        J_overint = LinearAlgebra.diagm(ones(length(r_overint)^dim)*dg.J[1]) #assume constant jacobian
    end
        
    if cmp(param.pde_type, "burgers1D")==0
        u_exact_overint = cos.(π*(x_overint.-current_time))
    else
        u_exact_overint = sin.(π * (x_overint.-current_time)) .+ 0.01
    end
    u_calc_final_overint = zeros(size(x_overint))
    u0_overint = zeros(size(x_overint))
    u_calc_final = zeros(dg.N_vol*dg.N_elem)
    for ielem = 1:dg.N_elem
        u_hat_local = zeros(length(dg.r_basis)^dim) 
        u0_hat_local = zeros(size(u_hat_local)) 
        for inode = 1:dg.Np
            u_hat_local[inode] = u_hat[dg.EIDLIDtoGID_basis[ielem,inode]]
            u0_hat_local[inode] = u_hat0[dg.EIDLIDtoGID_basis[ielem,inode]]
        end
        u_calc_final_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u_hat_local
        u0_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u0_hat_local
        u_calc_final[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol] .= dg.chi_v * u_hat_local
    end
    u_diff = u_calc_final_overint .- u_exact_overint

    x_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    u_calc_final_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    u0_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    u_exact_overint_1D = zeros(Np_overint_per_dim*dg.N_elem_per_dim)
    ctr = 1
    for iglobalID = 1:length(y_overint)
        if  y_overint[iglobalID] == 0
            x_overint_1D[ctr] = x_overint[iglobalID]
            u_calc_final_overint_1D[ctr] = u_calc_final_overint[iglobalID]
            u0_overint_1D[ctr] = u0_overint[iglobalID]
            u_exact_overint_1D[ctr] = u_exact_overint[iglobalID]
            ctr+=1
        end
    end

    L2_error::Float64 = 0
    energy_final_calc = 0
    energy_initial = 0
    for ielem = 1:dg.N_elem
        L2_error += (u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint]') * W_overint * J_overint * (u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint])
        
        # use non-overintegrated qties to calculate energy difference
        energy_final_calc += sum(((u_calc_final[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol]') * dg.W * dg.J * (u_calc_final[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol])))
        energy_initial += sum(((u0[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol]') * dg.W * dg.J * (u0[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol])))
    end
    L2_error = sqrt(L2_error)

    Linf_error = maximum(abs.(u_diff))

    energy_change = energy_final_calc - energy_initial

    Plots.vline(dg.VX, color="lightgray", linewidth=0.75, label="grid")
    Plots.plot!(vec(x_overint_1D), [vec(u_exact_overint_1D), vec(u_calc_final_overint_1D), vec(u0_overint_1D)], label=["exact" "calculated" "initial"])
    pltname = string("plt", N_elem_per_dim, ".pdf")
    Plots.savefig(pltname)
    
    #plot grid
    pltname = string("grid", N_elem_per_dim, ".pdf")
    Plots.vline(dg.VX, color="black", linewidth=0.75, label="grid")
    Plots.hline!(dg.VX, color="black", linewidth=0.75, label="grid")
    Plots.plot!(x_overint, y_overint, label=["nodes-overint"], seriestype=:scatter)
    Plots.plot!(dg.x, dg.y, label=["nodes"], seriestype=:scatter)
    Plots.savefig(pltname)


    if dim == 2 && N_elem_per_dim == 4
        PyPlot.figure("Initial cond, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u0_overint, 20)
        PyPlot.colorbar()
        Plots.savefig(pltname)
        PyPlot.figure("Final soln, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u_calc_final_overint, 20)
        PyPlot.colorbar()
        Plots.savefig(pltname)
    end

    return L2_error, Linf_error, energy_change#, solution
end

#==============================================================================
Global setup

Definition of domain
Discretize into elements
==============================================================================#

#main program
function main()

    # Polynomial order
    P = 3

    #N_elem_range = [4 8 16 32 64 128 256]# 512 1024]
    #N_elem_range = [2 4 8 16 32]
    N_elem_range = [2 4 8]# 16 32]
    #N_elem_range = [4]
    #N_elem_fine_grid = 1024 #fine grid for getting reference solution

    #_,_,reference_fine_grid_solution = setup_and_solve(N_elem_fine_grid,N)
    
    alpha_split = 1 #Discretization of conservative form
    #alpha_split = 2.0/3.0 #energy-stable split form
    
    dim=2
    fluxtype="split_with_LxF"
    #fluxtype="split"
    PDEtype = "burgers1D"
    #PDEtype = "linear_adv_1D"
    debugmode= false# if true, only solve one step using explicit Euler
    includesource = true
    volumenodes = "GL"
    basisnodes = "GLL"

    finaltime=0.50
    param = PhysicsAndFluxParams(dim, fluxtype, PDEtype, includesource, alpha_split, finaltime, volumenodes, basisnodes, debugmode)
    display(param)

    L2_err_store = zeros(length(N_elem_range))
    Linf_err_store = zeros(length(N_elem_range))
    energy_change_store = zeros(length(N_elem_range))

    for i=1:length(N_elem_range)

       #display("P = "*string(P))
       #display("N_elem_per_dim = "*string(N_elem_range[i]))
        # Number of elements
        N_elem =  N_elem_range[i]

        L2_err_store[i],Linf_err_store[i], energy_change_store[i] = setup_and_solve(N_elem,P,param)
    end

    Printf.@printf("P =  %d \n", P)

    dx = 2.0./N_elem_range

    Printf.@printf("n cells_per_dim    dx               L2 Error    L2  Error rate     Linf Error     Linf rate    Energy change \n")
    for i = 1:length(N_elem_range)
            conv_rate_L2 = 0.0
            conv_rate_Linf = 0.0
            if i>1
                conv_rate_L2 = log(L2_err_store[i]/L2_err_store[i-1]) / log(dx[i]/dx[i-1])
                conv_rate_Linf = log(Linf_err_store[i]/Linf_err_store[i-1]) / log(dx[i]/dx[i-1])
            end

            Printf.@printf("%d \t\t%.5f \t%.16f \t%.2f \t%.16f \t%.2f \t%.16f\n", N_elem_range[i], dx[i], L2_err_store[i], conv_rate_L2, Linf_err_store[i], conv_rate_Linf, energy_change_store[i])
    end
end

main()
