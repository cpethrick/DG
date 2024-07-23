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

function setup_and_solve(N_elem,P,param::PhysicsAndFluxParams)
    # N_elem is number of elements
    # N is poly order
    
    # Limits of computational domain
    x_Llim = 0.0
    x_Rlim = 2.0
    
    dim = 1
    #==============================================================================
    Start Up
    ==============================================================================#
    dg = init_DG(P, dim, N_elem, [x_Llim,x_Rlim])

    #plot grid
    pltname = string("grid", N_elem, ".pdf")
    Plots.vline(dg.VX, color="black", linewidth=0.75, label="grid")
    Plots.hline!(dg.VX, color="black", linewidth=0.75, label="grid")
    Plots.plot!(dg.x, dg.y, label=["nodes"], seriestype=:scatter)
    Plots.savefig(pltname)

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

    if param.include_source
        u0 = cos.(π * dg.x)
    else
        u0 = sin.(π * dg.x) .+ 0.01
    end
    if dim == 2
        PyPlot.clf()
        PyPlot.tricontourf(dg.x, dg.y, u0)
        Plots.savefig(pltname)
    end
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
    P = 2

    N_elem_range = [4 8 16 32 64 128 256]# 512 1024]
    #N_elem_range = [4]
    #N_elem_fine_grid = 1024 #fine grid for getting reference solution

    #_,_,reference_fine_grid_solution = setup_and_solve(N_elem_fine_grid,N)
    
    #alpha_split = 1 #Discretization of conservative form
    alpha_split = 2.0/3.0 #energy-stable split form
    
    param = PhysicsAndFluxParams("split", "burgers", true, alpha_split)

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
