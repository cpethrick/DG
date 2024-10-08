include("build_dg_residual.jl")
include("set_up_dg.jl")
include("parameters.jl")

import LinearMaps
import IterativeSolvers

function spacetimeimplicitsolve(u_hat0, dg::DG, param::PhysicsAndFluxParams)

    if cmp(param.spacetime_solver_type, "pseudotime")==0
        return pseudotimesolve(u_hat0, param.spacetime_decouple_slabs, dg, param)
    elseif cmp(param.spacetime_solver_type, "JFNK")==0
        return JFNKsolve(u_hat0, param.spacetime_decouple_slabs, dg, param)
    else
        display("Error: Space-time solver type is illegal!")
        return zeros(size(u_hat0))
    end


end

function JFNKsolve(u_hat0, do_decouple::Bool, dg::DG,param::PhysicsAndFluxParams)


    if do_decouple
        display("Decoupled PS")
        N_time_slabs = dg.N_elem_per_dim
    else
        display("Coupled PS")
        N_time_slabs = 1
    end

    u_hat = u_hat0
    for iTS = 1:N_time_slabs
        if do_decouple
            subset_EIDs = dg.TSIDtoEID[iTS,:]
            Printf.@printf("Time-slab ID: %d\n", iTS)
        else
            subset_EIDs = nothing
        end

        #========================
        # Meat of the solver here
        ========================#

        tol_NL = 1E-6
        tol_lin = 1E-10
        NL_iterlim = 100
        residual_NL = 1
        u_hat_NLiter = u_hat
        NL_iterctr = 0
        max_iterations = 2000
        if param.debugmode == true
            max_iterations= 1
        end
        #Outer loop: nonlinear iterations (Newton)
        while residual_NL > tol_NL && NL_iterctr < NL_iterlim 

            #Define function of only u_hat_in. Passing zero as time - not used in PS (as far as I recall).
            DG_residual_function(u_hat_in) =  assemble_residual(u_hat_in, 0.0, dg, param, subset_EIDs)
            perturbation = sqrt(eps())

            jacobian_vector_product(v) = 1/perturbation * ( DG_residual_function(u_hat_NLiter .+ perturbation * v) - DG_residual_function(u_hat_NLiter))

            

            FMap_DG_residual = LinearMaps.FunctionMap{Float64,false}(jacobian_vector_product, length(u_hat)) #second argument is size of the square linear map

            #Inner loop: linear iterations (GMRES - use package)
            u_hat_delta,log = IterativeSolvers.gmres!(u_hat,FMap_DG_residual, -1.0 * DG_residual_function(u_hat_NLiter); 
                                                     log=true, restart=500, abstol=tol_lin, reltol=tol_lin, verbose=false,
                                                     maxiter=max_iterations
                                                    ) #Note: gmres() initializes with zeros, while gmres!(x, FMap, b) initializes with x.)
            display(log)
            u_hat_NLiter += u_hat_delta
            residual_NL = sqrt(sum(u_hat_delta .^ 2))
            NL_iterctr+=1
            if param.debugmode
                residual_NL = 0.1*tol_NL
            end
            Printf.@printf("NL residual at iteration %d was %.3e\n", NL_iterctr, residual_NL)

        end
        u_hat = u_hat_NLiter

        u_hat0 = u_hat # Store u_hat in u_hat0 because the restarts use u_hat0. Unsure if this would be needed for JFNK

    end

    return u_hat
end


function pseudotimesolve(u_hat0, do_decouple::Bool, dg::DG, param::PhysicsAndFluxParams)

    if do_decouple
        display("Decoupled PS")
        N_time_slabs = dg.N_elem_per_dim
    else
        display("Coupled PS")
        N_time_slabs = 1
    end

    u_hat = u_hat0
    for iTS = 1:N_time_slabs
        if do_decouple
            subset_EIDs = dg.TSIDtoEID[iTS,:]
        else
            subset_EIDs = nothing
        end

        dt = 0.3* (dg.delta_x / dg.Np_per_dim) 
        residual = 1
        u_hat = u_hat0
        residual_scaling = sqrt(sum(u_hat.^2))
        first_residual = -1

        iterctr_all = 0

        # converge loosly with large time step
        while residual > 1E-5
            # solve a time step with RK
            # check difference between old and new solutions
            # compare to residual
            # if residual is decreasing, increase the CFL # to expidite solution
            (u_hatnew,current_time) = physicaltimesolve(u_hat, dt, 10, dg, param, subset_EIDs)
            u_change = u_hatnew - u_hat

            if param.debugmode
                residual = 1E-14
            else
            if first_residual < 0
                residual = sqrt(sum(u_change.^2))
                first_residual = residual
                display("First residual")
                display(first_residual)
            end
            residualnew = sqrt(sum(u_change.^2))/first_residual

            if isnan(residualnew)
                display("NaN detected, decreasing dt and restarting")
                u_hatnew = u_hat0
                residual=1
                residualnew=1
                dt *= 0.8
            end

            residual = residualnew
            #Printf.@printf("Residual =  %.3E \n", residual)
            u_hat = u_hatnew
            iterctr_all += 1

            if iterctr_all == 1000
                display("WARNING: not converged by 1000 iterations! Finishing the run.")
                residual = 1E-14
            end
        end
        end
        iterctr = 0
        # converge once more and decrease time step size every 100 iters
        if param.debugmode
            residual = 1E-14

        end
        while residual > 1E-12

            # solve a time step with RK
            # check difference between old and new solutions
            # compare to residual
            # if residual is decreasing, increase the CFL # to expidite solution
            (u_hatnew,current_time) = physicaltimesolve(u_hat, dt, 1, dg, param, subset_EIDs)
            u_change = u_hatnew - u_hat
            residualnew = sqrt(sum(u_change.^2))

            residual = residualnew
            #Printf.@printf("Residual =  %.3E \n", residual)
            u_hat = u_hatnew
            iterctr += 1
            iterctr_all +=1
            if iterctr > 50
                iterctr = 1
                dt *= 0.9
            end
        end

        Printf.@printf("Converged in %d iterations\n", iterctr_all)

        u_hat0 = u_hat # Store u_hat in u_hat0 because the restarts use u_hat0.

    end

    return u_hat
    
end


function physicaltimesolve(u_hat0, dt, Nsteps, dg, param, subset_EIDs=nothing)

    #==============================================================================
    RK scheme
    arguably should move the rk method out of this function in the future
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

    u_hat = u_hat0
    current_time = 0
    residual = zeros(size(u_hat))
    rhs = zeros(size(u_hat))
    for tstep = 1:Nsteps
        #for tstep = 1:1
        for iRKstage = 1:nRKStage

            rktime = current_time + rk4c[iRKstage] * dt

            #####assemble residual
            rhs = assemble_residual(u_hat, rktime, dg, param, subset_EIDs)

            residual = rk4a[iRKstage] * residual .+ dt * rhs
            u_hat += rk4b[iRKstage] * residual
        end
        current_time += dt   
    end

    return (u_hat, current_time)

end
