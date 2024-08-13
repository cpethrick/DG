include("build_dg_residual.jl")
include("physics.jl")
include("set_up_dg.jl")



function pseudotimesolve(u_hat0, dg::DG, param::PhysicsAndFluxParams)

    #initial dt
    dt = 0.1* (dg.delta_x / dg.Np_per_dim) 
    residual = 1
    u_hat = u_hat0
    residual_scaling = sqrt(sum(u_hat.^2))
    first_residual = -1
    while dt < 0.3*(dg.delta_x / dg.Np_per_dim)
        residual = 1
        while residual > 1E-5
           # solve a time step with RK
           # check difference between old and new solutions
           # compare to residual
           # if residual is decreasing, increase the CFL # to expidite solution
           u_hatnew = physicaltimesolve(u_hat, dt, 10, dg, param)
           u_change = u_hatnew - u_hat

           if first_residual < 0
               residual = sqrt(sum(u_change.^2))
               first_residual = residual
               display("First residual")
               display(first_residual)
           end
           residualnew = sqrt(sum(u_change.^2))/first_residual

           residual = residualnew
           display(residual)
           u_hat = u_hatnew
       end
       dt *= 2
       display("Increasing dt")
       display(dt)
   end
   iterctr = 0
   # converge once more
    while residual > 1E-12

       # solve a time step with RK
       # check difference between old and new solutions
       # compare to residual
       # if residual is decreasing, increase the CFL # to expidite solution
       u_hatnew = physicaltimesolve(u_hat, dt, 1, dg, param)
       u_change = u_hatnew - u_hat
       residualnew = sqrt(sum(u_change.^2))

       #if residualnew > residual
       #    dt*= 0.8
       #end

       residual = residualnew
       display(residual)
       u_hat = u_hatnew
       iterctr += 1
       if iterctr > 50
           iterctr = 1
           dt *= 0.9
       end
   end

    return u_hat
end


function physicaltimesolve(u_hat0, dt, Nsteps, dg, param)

    #==============================================================================
    RK scheme
    ==============================================================================#

    if true #param.debugmode == false
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
            rhs = assemble_residual(u_hat, rktime, dg, param)

            residual = rk4a[iRKstage] * residual .+ dt * rhs
            u_hat += rk4b[iRKstage] * residual
        end
        current_time += dt   
    end
    
    return u_hat

end
