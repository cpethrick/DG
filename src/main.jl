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
import FastGaussQuadrature

import SparseArrays
import LinearAlgebra
import Printf
import PyPlot
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
include("test_scripts.jl")


function calculate_projection_corrected_entropy_change(u_hat, dg::DG, param::PhysicsAndFluxParams)
    entropy_initial = 0
    entropy_final = 0
    projection_error = 0

    for ielem = 1:dg.N_elem
        if ielem < dg.N_elem_per_dim+1
            # face on bottom (t=0)
            
            # get initial condition
            x_local = dg.VX[ielem] .+ 0.5* (dg.r_flux.+1) * dg.delta_x
            y_local = zeros(size(x_local)) # leave zero as this is the lower face
            u_face_Dirichlet = calculate_solution_on_Dirichlet_boundary(x_local, y_local, dg,param)
            
            u_hat_local = u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof]
            u_face_interior = project(dg.chi_face[:,:,3],  u_hat_local,true,dg,param)


            s_vec = get_numerical_entropy_function(u_face_Dirichlet,param)
            entropy_initial += s_vec' * dg.W_face * dg.J_face * ones(size(s_vec))


            # get entropy potential and entropy variables at the face (interior soln)
            phi_face_interior = get_entropy_potential(u_face_interior, param)
            phi_face_Dirichlet = get_entropy_potential(u_face_Dirichlet, param)
            v_face_interior = get_entropy_variables(u_face_interior, param)
            v_face_Dirichlet = get_entropy_variables(u_face_Dirichlet, param)

            phi_jump = phi_face_interior - phi_face_Dirichlet
            v_jump = v_face_interior - v_face_Dirichlet


            vjumpTtimesu = zeros(size(phi_face_Dirichlet))
            for inode = 1:dg.N_vol_per_dim
                node_indices = dg.N_vol_per_dim*(1:dg.N_state) .- dg.N_vol_per_dim .+ inode
                vjumpTtimesu[inode] += (v_jump[node_indices])' * u_face_Dirichlet[node_indices]
            end


            projection_error += (phi_jump - vjumpTtimesu)' * dg.W_face * dg.J_face * ones(size(phi_jump))
        elseif ielem > dg.N_elem_per_dim^dg.dim - dg.N_elem_per_dim
            # face on top (t=tf)
            u_hat_local = u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof]
            u_face_interior = project(dg.chi_face[:,:,4],  u_hat_local,true,dg,param)

            s_vec = get_numerical_entropy_function(u_face_interior,param)
            entropy_final += s_vec' * dg.W_face * dg.J_face * ones(size(s_vec))
        end
    end

    display("Final entropy is ")
    display(entropy_final)
    display("Initial entropy from boundary condition is")
    display(entropy_initial)
    display("Projection error is ")
    display(projection_error)
    display("Entropy preservation:")
    display(entropy_final - entropy_initial + projection_error)
    display("Without projection correction:")
    display(entropy_final - entropy_initial)
    return entropy_final - entropy_initial + projection_error #return preservation

end

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
    else
        u_hat = spacetimeimplicitsolve(u_hat0 ,dg, param)
        #u_hat = u0
    end
    display("Reminder, c is ")
    display(param.fluxreconstructionC)
    #==============================================================================
    Analysis
    ==============================================================================#

    Np_overint_per_dim = dg.N_soln_per_dim+10
    Np_overint = (Np_overint_per_dim)^dim
    r_overint, w_overint = FastGaussQuadrature.gausslobatto(Np_overint_per_dim)
    (x_overint, y_overint) = build_coords_vectors(r_overint, dg)
    if dim==1
        chi_overint = vandermonde1D(r_overint,dg.r_basis)
        W_overint = LinearAlgebra.diagm(w_overint) # diagonal matrix holding quadrature weights
        J_overint = LinearAlgebra.diagm(ones(size(r_overint))*dg.J_soln[1]) #assume constant jacobian
    elseif dim==2
        chi_overint = vandermonde2D(r_overint, dg.r_basis, dg)
        W_overint = LinearAlgebra.diagm(vec(w_overint*w_overint'))
        J_overint = LinearAlgebra.diagm(ones(length(r_overint)^dim)*dg.J_soln[1]) #assume constant jacobian
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
    elseif cmp(param.pde_type, "euler1D")==0
        if param.usespacetime
            display("Warning: exact soln not correct for space time!")
            u_exact_overint_allstates = calculate_euler_exact_solution(-1.0, x_overint, y_overint, Np_overint, dg)
        else
            u_exact_overint_allstates = calculate_euler_exact_solution(current_time, x_overint, y_overint, Np_overint, dg)
        end
        # extract only density
        u_exact_overint = zeros(size(x_overint))
        for ielem = 1:dg.N_elem
            u_exact_overint[(1:Np_overint) .+ (ielem-1)*Np_overint]=u_exact_overint_allstates[ (1:Np_overint) .+ Np_overint*3*(ielem-1) .+ (1-1)*Np_overint]
        end
    else
        display("Warning - no exact solution defined!")
    end
    u_calc_final_overint = zeros(size(x_overint))
    u0_overint = zeros(size(x_overint))
    u_calc_final = zeros(dg.N_vol*dg.N_elem)
    for ielem = 1:dg.N_elem
        #Extract only first state here
        u_hat_local = zeros(length(dg.r_basis)^dim) 
        u0_hat_local = zeros(size(u_hat_local)) 
        for inode = 1:dg.N_soln
            # only istate = 1, which is velocity for lin adv or burgers
            # and density for euler
            u_hat_local[inode] = u_hat[dg.StIDGIDtoGSID[1,dg.EIDLIDtoGID_basis[ielem,inode]]]
            u0_hat_local[inode] = u_hat0[dg.StIDGIDtoGSID[1,dg.EIDLIDtoGID_basis[ielem,inode]]]
        end
        u_calc_final_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u_hat_local
        u0_overint[(ielem-1)*Np_overint+1:(ielem)*Np_overint] .= chi_overint * u0_hat_local
        u_calc_final[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol] .= dg.chi_soln* u_hat_local
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
    if param.usespacetime
        # initial is 0.0, final is 2.0
        ctr0 = 1
        ctr2 = 1
        for iglobalID = 1:length(y_overint)
            if  y_overint[iglobalID] == 2.0
                u_calc_final_overint_1D[ctr2] = u_calc_final_overint[iglobalID]
                u_exact_overint_1D[ctr2] = u_exact_overint[iglobalID]
                ctr2+=1
            elseif y_overint[iglobalID] == 0.0
                u0_overint_1D[ctr0] = u_calc_final_overint[iglobalID]
                ctr0+=1
            end
        end
    end

    L2_error::Float64 = 0
    entropy_final_calc = 0
    entropy_initial = 0
    for ielem = 1:dg.N_elem
        L2_error += (u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint]') * W_overint * J_overint * (u_diff[(ielem-1)*Np_overint+1:(ielem)*Np_overint])

        if param.usespacetime
            # for spacetime, we want to consider initial as t=0 (bottom surface of the computational domain) and final as t=t_f (top surface)

            if ielem < N_elem_per_dim+1
                # face on bottom
                #u_face = dg.chi_f[:,:,3] * u_hat[(ielem-1)*dg.N_vol+1:(ielem)*dg.N_vol]
                

                # Find initial energy from the Dirichlet BC on lower face
                x_local = dg.VX[ielem] .+ 0.5* (dg.r_flux.+1) * dg.delta_x
                y_local = zeros(size(x_local)) # leave zero as this is the lower face
                u_face = calculate_solution_on_Dirichlet_boundary(x_local, y_local,dg, param)
                S_face = get_numerical_entropy_function(u_face, param)

                entropy_initial += S_face' * dg.W_face * dg.J_face * ones(size(S_face))
            elseif ielem > N_elem_per_dim^dim - N_elem_per_dim
                # face on top
                u_face = project(dg.chi_face[:,:,4], u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof], true, dg, param)
                S_face = get_numerical_entropy_function(u_face, param)
                entropy_final_calc += S_face' * dg.W_face * dg.J_face * ones(size(S_face))
            end
            if param.fluxreconstructionC > 0
                display("WARNING: Energy calculation is probably unreliable for c != 0.")
            end
        else
            entropy_final_calc += calculate_integrated_numerical_entropy(u_hat[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof], dg, param)
            entropy_initial += calculate_integrated_numerical_entropy(u_hat0[(ielem-1)*dg.N_soln_dof+1:(ielem)*dg.N_soln_dof], dg, param)
        end
    end

    L2_error = sqrt(L2_error)

    Linf_error = maximum(abs.(u_diff))

    entropy_change = entropy_final_calc - entropy_initial

    if param.usespacetime
        proj_corrected_error = calculate_projection_corrected_entropy_change(u_hat, dg, param)
        entropy_change = proj_corrected_error
    end
    PyPlot.figure("Solution", figsize=(6,4))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks(dg.VX, minor=false)
    ax.xaxis.grid(true, which="major")
    PyPlot.plot(vec(x_overint_1D), vec(u0_overint_1D), label="initial")
    if param.include_source || cmp(param.pde_type, "linear_adv_1D")==0
        PyPlot.plot(vec(x_overint_1D), vec(u_exact_overint_1D), label="exact")
    end
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

    for i in 1:length(u_calc_final_overint)
        if isnan(u_calc_final_overint[i])
            u_calc_final_overint[i]=0
        end
    end

    if dim == 2# && N_elem_per_dim == 4
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
    return L2_error, Linf_error, entropy_change#, solution
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
        (L2_err_store[i],Linf_err_store[i], entropy_change_store[i]) = setup_and_solve(N_elem,P,param)
        # End timer
        time_store[i] = time() - t

        if cmp(param.pde_type, "euler1D")==0
            display("Convergence is for density. Energy is probably not valid.")
        end

        #Evalate convergence, print, and save to file
        Printf.@printf("P =  %d \n", P)
        dx = 2.0./N_elem_range
        Printf.@printf("n cells_per_dim    dx               L2 Error    L2  Error rate     Linf Error     Linf rate    Entropy change         Time   Time scaling\n")
        if cmp(param.convergence_table_name, "none") != 0
            fname = "result/"*param.convergence_table_name*".csv" #Note: this should be changed to something useful in the future...
            f = open(fname, "w")
            DelimitedFiles.writedlm(f, ["n cells_per_dim" "dx" "L2 Error" "L2  Error rate" "Linf Error" "Linf rate" "Entropy change" "Time" "Time scaling"], ",")
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
                DelimitedFiles.writedlm(f, [N_elem_range[j], dx[j], L2_err_store[j], conv_rate_L2, Linf_err_store[j], conv_rate_Linf, entropy_change_store[j    ], time_store[j], conv_rate_time]', ",")
            end

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
