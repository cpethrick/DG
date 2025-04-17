#==============================================================================
Import packages
==============================================================================#

# For Jacobi/Legendre polynomials
# E.g., ClassicalOrthagonalPolynomials.jacobip(N, alpha, beta, x)
# NOTE: Need to normalize manually
# import ClassicalOrthogonalPolynomials

# For Gauss quadrature points
# E.g., xi,wi = FastGaussQuadrature.gaussjacobi(N,0.0,0.0) is Gauss-Legendre nodes
#  Returns ([xi],[wi])
# E.g., FastGaussQuadrature.gausslobatto(N) is Gauss-Legendre-Lobatto nodes
#import FastGaussQuadrature

import SparseArrays
import LinearAlgebra
import Printf
import DelimitedFiles

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
include("parameters.jl")
include("post_processing.jl")
include("test_scripts.jl")
include("cost_tracking.jl")


function calculate_integrated_numerical_entropy(u_hat, dg::DG, param::PhysicsAndFluxParams)
    
    if cmp(param.pde_type, "euler1D")==0
        u = project(dg.chi_soln,u_hat,false,dg, param)
        s = get_numerical_entropy_function(u, param)
        return s' * dg.W_soln * dg.J_soln * ones(size(s))
    else
        return u_hat' * dg.M * u_hat
    end
end

function setup_and_solve(N_elem_per_dim,P,param::PhysicsAndFluxParams)
    # N_elem_per_dim is number of elements PER DIM
    # N is poly order

    # Limits of computational domain
    x_Llim = 0.0
    x_Rlim = param.domain_size

    dim = param.dim 
    #==============================================================================
    Start Up
    ==============================================================================#
    if cmp(param.pde_type, "euler1D") == 0
        N_state = 3
    else
        N_state = 1
    end
    dg = init_DG(param.P, param.dim, N_elem_per_dim, N_state, [x_Llim,x_Rlim], param.volumenodes, param.basisnodes, param.fluxnodes, param.fluxnodes_overintegration, param.fluxreconstructionC, param.usespacetime)

    cost_tracker = init_CostTracking()

    #==============================================================================
    Initialization
    ==============================================================================#

    finaltime = param.finaltime

    u0 = calculate_initial_solution(dg, param)
    u_hat0 = zeros(dg.N_soln_dof_global)
    u_local_state = zeros(dg.N_soln)
    for ielem = 1:dg.N_elem
        for istate = 1:dg.N_state
            for inode = 1:dg.N_vol
                u_local_state[inode] = u0[dg.StIDGIDtoGSID[istate,dg.EIDLIDtoGID_vol[ielem,inode]]]
            end
            u_hat_local_state = dg.Pi_soln*u_local_state
            u_hat0[dg.StIDGIDtoGSID[istate,dg.EIDLIDtoGID_basis[ielem,:]]] = u_hat_local_state
        end
    end

    #==============================================================================
    ODE Solver 
    ==============================================================================#
    if !param.usespacetime
        #Physical time
        #timestep size according to CFL
        CFL = 0.05
        #xmin = minimum(abs.(x[1,:] .- x[2,:]))
        #dt = abs(CFL / a * xmin /2)
        dt = CFL * (dg.delta_x / dg.N_soln_per_dim)
        Nsteps::Int64 = ceil(finaltime/dt)
        dt = finaltime/Nsteps
        if param.debugmode == true
            Nsteps = 1
        end
        display("Beginning time loop")
        (u_hat,current_time) = physicaltimesolve(u_hat0, dt, Nsteps, dg, param)
        display("Done time loop")
        L2_error, Linf_error, entropy_change = post_process(u_hat, current_time, u_hat0, dg, param) 
    elseif param.read_soln_from_file
        # user is responsible for ensuring that this file exists...
        u_hat = vec(DelimitedFiles.readdlm("u_hat_stored.csv"))
        L2_error, Linf_error, entropy_change = post_process(u_hat, u_hat0, dg, param) 
    else
        u_hat = spacetimeimplicitsolve(u_hat0, dg, param, cost_tracker)
        write_to_file(u_hat)
        #== uncomment to write solution to file
       
        write_to_file(u_hat)
        
        ==#
        #u_hat = u0
        L2_error, Linf_error, entropy_change = post_process(u_hat, u_hat0, dg, param) 
    end
    display("Reminder, c is ")
    display(param.fluxreconstructionC)

    summary(cost_tracker)

    return L2_error, Linf_error, entropy_change, cost_tracker#, solution
end

function run(param::PhysicsAndFluxParams)


    P = param.P
    N_elem_range = 2 .^(1:param.n_times_to_solve)
    L2_err_store = zeros(length(N_elem_range))
    Linf_err_store = zeros(length(N_elem_range))
    entropy_change_store = zeros(length(N_elem_range))
    time_store = zeros(length(N_elem_range))

    for i=1:length(N_elem_range)

        # Number of elements
        N_elem =  N_elem_range[i]

        # Start timer (NOTE: should change to BenchmarkTools.jl in the future)
        t = time()
        # Solve
        (L2_err_store[i],Linf_err_store[i], entropy_change_store[i], cost_tracker) = setup_and_solve(N_elem,P,param)
        # End timer
        time_store[i] = time() - t

        if cmp(param.pde_type, "euler1D")==0
            display("Convergence is for density.")
        end

        #Evalate convergence, print, and save to file
        Printf.@printf("P =  %d \n", P)
        dx = 2.0./N_elem_range
        Printf.@printf("n cells_per_dim    dx               L2 Error    L2  Error rate     Linf Error     Linf rate    Entropy change         Time   Time scaling\n")
        if cmp(param.convergence_table_name, "none") != 0
            fname = "result/"*param.convergence_table_name*".csv" #Note: this should be changed to something useful in the future...
            f = open(fname, "w")
            DelimitedFiles.writedlm(f, ["n cells_per_dim" "dx" "L2 Error" "L2  Error rate" "Linf Error" "Linf rate" "Entropy change" "Time" "Time scaling"], ",")
        end
        for j = 1:i
            conv_rate_L2 = 0.0
            conv_rate_Linf = 0.0
            conv_rate_time = 0.0
            conv_rate_energy = 0.0
            if j>1
                conv_rate_L2 = log(L2_err_store[j]/L2_err_store[j-1]) / log(dx[j]/dx[j-1])
                conv_rate_Linf = log(Linf_err_store[j]/Linf_err_store[j-1]) / log(dx[j]/dx[j-1])
                conv_rate_time = log(time_store[j]/time_store[j-1]) / log(dx[j]/dx[j-1])
                conv_rate_energy = log(abs(entropy_change_store[j]/entropy_change_store[j-1])) / log(dx[j]/dx[j-1])
            end
            Printf.@printf("%d \t\t%.5f \t%.16f \t%.2f \t%.16f \t%.2f \t%.4e \t%.2f \t%.5e \t%.2f\n", N_elem_range[j], dx[j], L2_err_store[j], conv_rate_L2, Linf_err_store[j], conv_rate_Linf, entropy_change_store[j], conv_rate_energy, time_store[j], conv_rate_time)
            if cmp(param.convergence_table_name, "none") != 0
                DelimitedFiles.writedlm(f, [N_elem_range[j], dx[j], L2_err_store[j], conv_rate_L2, Linf_err_store[j], conv_rate_Linf, entropy_change_store[j    ], time_store[j], conv_rate_time]', ",")
            end

        end

        if cmp(param.convergence_table_name, "none") != 0
            close(f)
        end

    end
end
#==============================================================================
Global setup

Definition of domain
Discretize into elements
==============================================================================#

#main program
function main(paramfile::AbstractString="default_parameters.csv")
    #==
    # Polynomial order
    P = 2

    # Range of element numbers to solve.
    # Use an array for a refinement study, e.g. N_elem_range = [2 4 8 16 32]
    # Or a single value, e.g. N_elem_range = [3]
    N_elem_range = [2 4 8 16 32 64]

    # Dimension of the grid.
    # Can be 1 or 2.
    dim=1

    # PDE type to solve. 
    # "burgers1D" will solve 1D burgers on a 1D grid or 1D burgers on a 2D grid with no flux in the y-direction.
    # "linear_adv_1D" will solve 1D linear advection in the x-direction with specified velocity on a 1D or 2D grid.
    # "burgers2D" will solve 2D burgers on a 2D grid.
    PDEtype = "burgers1D"

    # Toggle for whether to use space-time.
    # Should set dim=2 and use "linear_adv_1D" PDE.
    usespacetime = false

    # Type of flux to use for the numerical flux.
    # If the PDE type is linear_adv_1D, pure upwinding will ALWAYS be chosen.
    # Either choice split or split_with_LxF will result in upwinding.
    # You can hard-code central in physics.jl.
    # If the PDE type is burgers, "split" is a pure energy-conserving flux
    # and "split_with_LxF" adds LxF upwinding to the energy-conserving flux.
    fluxtype="split"

    # Relative weighting of conservative and non-conservative forms
    # alpha_split=1 recovers the conservative discretization.
    # alpha_split = 2.0/3.0 will be energy-conservative for Burgers' equation.
    alpha_split = 2.0/3.0

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
    P=4
    cp = factorial(2*P) / (2^P * factorial(P)^2)
    # c-
    #fluxreconstructionC = -1/((2 * P + 1) * ( factorial(P) * cp )^2)^dim 
    # cSD
    fluxreconstructionC = P /((P+1) * ((2 * P + 1) * ( factorial(P) * cp )^2)^dim)
    # cHU
    #fluxreconstructionC = (P+1) /((P) * ( (2 * P + 1) * ( factorial(P) * cp )^2)^dim)
    # cPlus for P=2, RK44 from Castonguay 2012 thesis
    #fluxreconstructionC = 0.183 / 2 # divide by 2 for normalized Legendre basis
    # cPlus for P=3, RK44
    #fluxreconstructionC = 3.60E−3 / 2
    fluxreconstructionC = 1E-8

    # Include a manufactured solution source.
    # For 1D burgers on any grid, this will add the manufactured solution required to 
    # perform a grid refinement study and find OOAs.
    # 2D Burgers manufactured solution is not yet implemented.
    # The initial condition is set based on the inclusion of a source.
    includesource = false

    # FInal time to run the simulation for.
    # Solves with RK4.
    finaltime= 0.2 # space-time: use at least 4 to allow enough time for information to propagate through the domain times 2

    # Run in debug mode.
    # if true, only solve one step using explicit Euler, ignoring finaltime.
    debugmode = false

    #Pack parameters into a struct
    param = PhysicsAndFluxParams(dim, fluxtype, PDEtype, usespacetime, includesource, alpha_split, advection_speed, finaltime, volumenodes, basisnodes, fluxreconstructionC, debugmode)
    display(param)
    ==#
    param = parse_parameters(paramfile)

    run(param)
end

main()
main("spacetime_euler_entropy_preservation.csv")
