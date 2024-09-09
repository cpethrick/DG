include("build_dg_residual.jl")
include("set_up_dg.jl")
include("parameters.jl")

function pseudotimesolve_decoupled(u_hat0, dg::DG, param::PhysicsAndFluxParams)
    # Pseudotime solve while decoupling the time-slabs.
    # for now, copy-paste from other function mostly. That is uneligant and shoud be fixed.

    display("Decoupled PS")

    u_hat = u_hat0
    for iTS=1:dg.N_elem_per_dim
        subset_EIDs = dg.TSIDtoEID[iTS,:]
    
        ########################### COPY STARTS HERE
        #initial dt
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
            if iterctr > 50
                iterctr = 1
                dt *= 0.9
            end
        end
        ######################3 COPY ENDS HERE

        u_hat0 = u_hat # Store u_hat in u_hat0 because the restarts use u_hat0.
    end

    return u_hat
end

function pseudotimesolve(u_hat0, dg::DG, param::PhysicsAndFluxParams)

    display("Normal PS")

    #initial dt
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
        (u_hatnew,current_time) = physicaltimesolve(u_hat, dt, 10, dg, param)
        u_change = u_hatnew - u_hat

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

        if param.debugmode
            # setting residual = 1E-14 ensures that we only do 1 time step
            residual = 1E-14
        end
    end
    iterctr = 0
    # converge once more and decrease time step size every 100 iters
    while residual > 5E-16

        # solve a time step with RK
        # check difference between old and new solutions
        # compare to residual
        # if residual is decreasing, increase the CFL # to expidite solution
        (u_hatnew,current_time) = physicaltimesolve(u_hat, dt, 1, dg, param)
        u_change = u_hatnew - u_hat
        residualnew = sqrt(sum(u_change.^2))

        residual = residualnew
        #Printf.@printf("Residual =  %.3E \n", residual)
        u_hat = u_hatnew
        iterctr += 1
        if iterctr > 50
            iterctr = 1
            dt *= 0.9
        end
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
