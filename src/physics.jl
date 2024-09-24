#==============================================================================
# Functions specific to the problem's physics
==============================================================================#

include("set_up_dg.jl")
include("parameters.jl")
function transform_physical_to_reference(f_physical, direction, dg::DG)
    return dg.C_m[direction,direction] * f_physical 
end

function calculate_numerical_flux(uM_face,uP_face,n_face, direction,dg::DG, param::PhysicsAndFluxParams)
    f_numerical=zeros(size(uM_face))

    if direction == 2 && param.usespacetime 
        # f_numerical = 0.5 * ( uM_face + uP_face )
        # second direction corresponding to time.
        # only use one-sided information such that the flow of information is from past to future.
        #
        # NOTE: must disable the decouple time slabs option if using a flux other than pure upwinding!
        if n_face[direction] == -1
            # face is bottom. Use the information from the external element
            # which corresponds to the past
            f_numerical = uP_face
        elseif n_face[direction] == 1
            # face is bottom. Use internal solution
            # which corresonds to the past
            f_numerical = uM_face
        end
    elseif cmp(param.pde_type, "linear_adv_1D")==0
        a = param.advection_speed
        alpha = 0 #upwind
        #alpha = 1 #central
        if direction ==1 
            f_numerical = 0.5 * a * (uM_face .+ uP_face) .+ a * (1-alpha) / 2.0 * (n_face[direction]) * (uM_face.-uP_face) # lin. adv, upwind/central
        end # numerical flux only in x-direction.
    elseif cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        f_numerical  = 1.0/6.0 * (uM_face .* uM_face + uM_face .* uP_face + uP_face .* uP_face) # split
        if cmp(param.numerical_flux_type, "split_with_LxF")==0
            stacked_MP = [uM_face;uP_face]
            max_eigenvalue = findmax(abs.(stacked_MP))[1]
            f_numerical += n_face[direction] * 0.5 .* max_eigenvalue .* (uM_face .- uP_face)
        end
    end

    if dg.dim == 2
        # in 1D, C_m = 1 so we don't need this step
        f_numerical = transform_physical_to_reference(f_numerical, direction, dg)
    end
    return f_numerical

end

function calculate_flux(u, direction, dg::DG, param::PhysicsAndFluxParams)
    f = zeros(dg.N_vol)

    if direction == 2 && param.usespacetime
        f .+= u
    elseif cmp(param.pde_type,"linear_adv_1D")==0
        if direction == 1
            f .+= param.advection_speed .* u # nodal flux for lin. adv.
        end
    elseif cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        f += 0.5 .* (u.*u) # nodal flux
    end

    #display("f_physical")
    #display(f)
    if dg.dim == 2
        # in 1D, C_m = 1 so we don't need this step
        f = transform_physical_to_reference(f, direction, dg)
        #display(f)
    end
    f_hat = dg.Pi * f

    return f_hat#,f_f

end

function calculate_face_terms_nonconservative(chi_face, u_hat, direction, dg::DG, param::PhysicsAndFluxParams)
    if cmp(param.pde_type,"burgers2D")==0 || (cmp(param.pde_type,"burgers1D")==0 && direction == 1)
        u_physical = chi_face*u_hat
        f_physical =  0.5 * (u_physical) .* (u_physical)
        f_reference = transform_physical_to_reference(f_physical, direction, dg)
    else
        return 0*(chi_face*u_hat)
    end
    return f_reference
end 

function calculate_initial_solution(x::AbstractVector{Float64},y::AbstractVector{Float64}, param::PhysicsAndFluxParams)


    if param.usespacetime
        #u0 = cos.(π * (x))
        u0 = 0*x
    elseif param.include_source && cmp(param.pde_type, "burgers2D")==0
        u0 = cos.(π * (x + y))
    elseif param.include_source && cmp(param.pde_type, "burgers1D")==0
        u0 = cos.(π * (x))
    elseif cmp(param.pde_type, "burgers2D") == 0
        u0 = exp.(-10*((x .-1).^2 .+(y .-1).^2))
    else
        #u0 = 0.2* sin.(π * x) .+ 0.01
        u0 = sin.(π * (x)) .+ 0.01
    end
    return u0
end

function calculate_source_terms(x::AbstractVector{Float64},y::AbstractVector{Float64},t::Float64, param::PhysicsAndFluxParams)
    if param.include_source
        if cmp(param.pde_type, "linear_adv_1D") ==0
            display("Warning! You probably don't want the source for linear advection!")
        end
        if cmp(param.pde_type, "burgers1D")==0 && param.usespacetime
            # y is time
            return π*sin.(π*(x .- y)).*(1 .- cos.(π*(x - y)))
        elseif cmp(param.pde_type, "burgers1D")==0
            return π*sin.(π*(x .- t)).*(1 .- cos.(π*(x .- t)))
        elseif cmp(param.pde_type, "burgers2D")==0 
            display("Warning! This source is not correct!")
            return π*sin.(π*(x .+ y.- sqrt(2) * t)).*(1 .- cos.(π*(x .+ y .- sqrt(2) * t)))
        end
    else
        return zeros(size(x))
    end
end

function calculate_solution_on_Dirichlet_boundary(x::AbstractVector{Float64},y::AbstractVector{Float64}, param::PhysicsAndFluxParams)

    if param.include_source
        return  cos.(π * (x-y))
    elseif cmp(param.pde_type, "burgers1D")==0
        return 0.2*sin.(π * (x .- 0.314159265359878323))
        out = 0.2* ones(size(x))
        first1 = true # double-valued at 1.0 if using GLL, so force it to set 0 on left side
        for i in 1:length(x)
            xcoord = x[i]
            if xcoord == 1.0 && first1
                first1=false
            elseif xcoord >=  1.0
                out[i] = 0
            end
        end
        return out

    else
        return sin.(π * (x)) .+ 0.01
    end
end
