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
import PyPlot


#==============================================================================
Handy function to avoid unintentionally exiting :)
==============================================================================#
function myexit()
    Printf.@printf("Do you really want to exit? [y/n] \n")
    exitresponse::String = readline()

    if cmp(exitresponse,"y")==0
        exit()
    else
        Printf.@printf("Not exiting for now.")
    end
end

#==============================================================================
Load external files
==============================================================================#

include("physics.jl")
include("FE_basis.jl")
include("FE_mapping.jl")
include("build_dg_residual.jl")
include("set_up_dg.jl")
include("ode_solver.jl")

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
    dg = init_DG(P, dim, N_elem_per_dim, [x_Llim,x_Rlim], param.volumenodes, param.basisnodes, param.fluxreconstructionC, param.usespacetime)

    #==============================================================================
    Initialization
    ==============================================================================#

    finaltime = param.finaltime

    u0 = calculate_initial_solution(dg.x, dg.y, param)
    u_hat0 = zeros(dg.N_elem*dg.Np)
    u_local = zeros(dg.N_vol)
    for ielem = 1:dg.N_elem
        for inode = 1:dg.N_vol
            u_local[inode] = u0[dg.EIDLIDtoGID_vol[ielem,inode]]
        end
        u_hat_local = dg.Pi*u_local
        u_hat0[dg.EIDLIDtoGID_basis[ielem,:]] = u_hat_local
    end

    #==============================================================================
    ODE Solver 
    ==============================================================================#
    if !param.usespacetime
        #Physical time
        #timestep size according to CFL
        CFL = 0.005
        #xmin = minimum(abs.(x[1,:] .- x[2,:]))
        #dt = abs(CFL / a * xmin /2)
        dt = CFL * (dg.delta_x / dg.Np_per_dim)
        Nsteps::Int64 = ceil(finaltime/dt)
        dt = finaltime/Nsteps
        if param.debugmode == true
            Nsteps = 1
        end
        
        (u_hat,current_time) = physicaltimesolve(u_hat0, dt, Nsteps, dg, param)
    else
        u_hat = pseudotimesolve(u_hat0, dg, param)
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
        
    if cmp(param.pde_type, "burgers1D")==0 && param.usespacetime
        # y is time
        u_exact_overint = cos.(π*(x_overint-y_overint))
    elseif cmp(param.pde_type, "burgers1D")==0
        u_exact_overint = cos.(π*(x_overint.-current_time))
    elseif cmp(param.pde_type, "burgers2D")==0
        u_exact_overint = cos.(π*(x_overint.+y_overint.-sqrt(2)*current_time))
    elseif cmp(param.pde_type, "linear_adv_1D")==0 && param.usespacetime == false
        u_exact_overint = sin.(π * (x_overint.- param.advection_speed * current_time)) .+ 0.01
    elseif cmp(param.pde_type, "linear_adv_1D")==0 && param.usespacetime == true 
        u_exact_overint = sin.(π * (x_overint - param.advection_speed * y_overint)) .+ 0.01
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
        if  y_overint[iglobalID] == 0.0
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

    PyPlot.figure("Solution", figsize=(6,4))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks(dg.VX, minor=true)
    ax.xaxis.grid(true, which="major")
    PyPlot.plot(vec(x_overint_1D), vec(u0_overint_1D), label="initial")
    PyPlot.plot(vec(x_overint_1D), vec(u_exact_overint_1D), label="exact")
    PyPlot.plot(vec(x_overint_1D), vec(u_calc_final_overint_1D), label="calculated")
    #Plots.plot!(vec(x_overint_1D), [vec(u_calc_final_overint_1D), vec(u0_overint_1D)], label=["calculated" "initial"])
    PyPlot.legend()
    pltname = string("plt", N_elem_per_dim, ".pdf")
    PyPlot.savefig(pltname)
    
    PyPlot.figure("Grid", figsize=(6,6))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks(dg.VX, minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    if dim == 1
        PyPlot.axhline(0)
    elseif dim == 2
        ax.set_yticks(dg.VX, minor=false)
        ax.yaxis.grid(true, which="major", color="k")
    end
    PyPlot.plot(x_overint, y_overint,"o", color="yellowgreen", label="overintegration", markersize=0.25)
    PyPlot.plot(dg.x, dg.y, "o", color="darkolivegreen", label="volume nodes")
    pltname = string("grid", N_elem_per_dim, ".pdf")
    PyPlot.savefig(pltname)
        
    
    if dim == 2 && N_elem_per_dim == 4
        PyPlot.figure("Initial cond, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u0_overint, 20)
        PyPlot.colorbar()
        PyPlot.figure("Final soln, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u_calc_final_overint, 20)
        PyPlot.colorbar()
        PyPlot.figure("Final exact soln, overintegrated")
        PyPlot.clf()
        PyPlot.tricontourf(x_overint, y_overint, u_exact_overint, 20)
        PyPlot.colorbar()
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
    P = 2

    # Range of element numbers to solve.
    # Use an array for a refinement study, e.g. N_elem_range = [2 4 8 16 32]
    # Or a single value, e.g. N_elem_range = [3]
    N_elem_range = [2 4 8 16]
   
    # Dimension of the grid.
    # Can be 1 or 2.
    dim=2
    
    # PDE type to solve. 
    # "burgers1D" will solve 1D burgers on a 1D grid or 1D burgers on a 2D grid with no flux in the y-direction.
    # "linear_adv_1D" will solve 1D linear advection in the x-direction with specified velocity on a 1D or 2D grid.
    # "burgers2D" will solve 2D burgers on a 2D grid.
    PDEtype = "linear_adv_1D"

    # Toggle for whether to use space-time.
    # Should set dim=2 and use "linear_adv_1D" PDE.
    usespacetime = true

    # Type of flux to use for the numerical flux.
    # If the PDE type is linear_adv_1D, pure upwinding will ALWAYS be chosen.
    # Either choice split or split_with_LxF will result in upwinding.
    # You can hard-code central in physics.jl.
    # If the PDE type is burgers, "split" is a pure energy-conserving flux
    # and "split_with_LxF" adds LxF upwinding to the energy-conserving flux.
    fluxtype="split_with_LxF"

    # Relative weighting of conservative and non-conservative forms
    # alpha_split=1 recovers the conservative discretization.
    # alpha_split = 2.0/3.0 will be energy-conservative for Burgers' equation.
    alpha_split = 1.0 

    # Advection speed
    advection_speed = 0.5
    
    # Choice of nodes for volume and basis nodes.
    # Options are "GL" for Gauss-Legendre or "GLL" for Gauss-Legendre-Lobatto.
    # Any combination should work.
    volumenodes = "GL"
    basisnodes = "GLL"


    # Flux reconstruction parameter "c"
    # In normalized Legendre reference basis per Table 1 of Cicchino 2021
    #
    # cDG
    fluxreconstructionC = 0 
    P=3
    cp = factorial(2*P) / (2^P * factorial(P)^2)
    # c-
    #fluxreconstructionC = -1/((2 * P + 1) * ( factorial(P) * cp )^2)^dim 
    # cSD
    #fluxreconstructionC = P /((P+1) * ((2 * P + 1) * ( factorial(P) * cp )^2)^dim)
    # cHU
    fluxreconstructionC = (P+1) /((P) * ( (2 * P + 1) * ( factorial(P) * cp )^2)^dim)
    # cPlus for P=2, RK44 from Castonguay 2012 thesis
    #fluxreconstructionC = 0.183 / 2 # divide by 2 for normalized Legendre basis
    # cPlus for P=3, RK44
    #fluxreconstructionC = 3.60E−3 / 2
    # fluxreconstructionC = 0.1/2

    # Include a manufactured solution source.
    # For 1D burgers on any grid, this will add the manufactured solution required to 
    # perform a grid refinement study and find OOAs.
    # 2D Burgers manufactured solution is not yet implemented.
    # The initial condition is set based on the inclusion of a source.
    includesource = false

    # FInal time to run the simulation for.
    # Solves with RK4.
    finaltime= 1 # space-time: use at least 4 to allow enough time for information to propagate through the domain times 2
    
    # Run in debug mode.
    # if true, only solve one step using explicit Euler, ignoring finaltime.
    debugmode = false

    #Pack parameters into a struct
    param = PhysicsAndFluxParams(dim, fluxtype, PDEtype, usespacetime, includesource, alpha_split, advection_speed, finaltime, volumenodes, basisnodes, fluxreconstructionC, debugmode)
    display(param)

    L2_err_store = zeros(length(N_elem_range))
    Linf_err_store = zeros(length(N_elem_range))
    energy_change_store = zeros(length(N_elem_range))

    for i=1:length(N_elem_range)

        # Number of elements
        N_elem =  N_elem_range[i]

        #Solve
        L2_err_store[i],Linf_err_store[i], energy_change_store[i] = setup_and_solve(N_elem,P,param)

        #Evalate convergence and print
        Printf.@printf("P =  %d \n", P)
        dx = 2.0./N_elem_range
        Printf.@printf("n cells_per_dim    dx               L2 Error    L2  Error rate     Linf Error     Linf rate    Energy change \n")
        for j = 1:i
                conv_rate_L2 = 0.0
                conv_rate_Linf = 0.0
                if j>1
                    conv_rate_L2 = log(L2_err_store[j]/L2_err_store[j-1]) / log(dx[j]/dx[j-1])
                    conv_rate_Linf = log(Linf_err_store[j]/Linf_err_store[j-1]) / log(dx[j]/dx[j-1])
                end
                Printf.@printf("%d \t\t%.5f \t%.16f \t%.2f \t%.16f \t%.2f \t%.16f\n", N_elem_range[j], dx[j], L2_err_store[j], conv_rate_L2, Linf_err_store[j], conv_rate_Linf, energy_change_store[j])
        end

    end
end

main()
