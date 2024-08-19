#==============================================================================
# Functions specific to the problem's physics
# Currently solves Linear Advection.
==============================================================================#

import CSV
import DataFrames

include("set_up_dg.jl")

mutable struct PhysicsAndFluxParams
    dim::Int64
    n_times_to_solve::Int64
    P::Int64
    numerical_flux_type::AbstractString
    pde_type::AbstractString
    usespacetime::Bool
    include_source::Bool
    alpha_split::Float64
    advection_speed::Float64
    finaltime::Float64
    volumenodes::String #"GLL" or "GL"
    basisnodes::String #"GLL" or "GL"
    fr_c_name::String # cDG, cPlus, cHU, cSD, c-, 1000
    debugmode::Bool

    # dependant params: set based on above required params.
    #set based on value of fr_c_name
    fluxreconstructionC::Float64


    # incomplete initialization: leave dependant variables uninitialized.
    PhysicsAndFluxParams(
                        dim::Int64,
                        n_times_to_solve::Int64,
                        P::Int64,
                        numerical_flux_type::AbstractString,
                        pde_type::AbstractString,
                        usespacetime::Bool,
                        include_source::Bool,
                        alpha_split::Float64,
                        advection_speed::Float64,
                        finaltime::Float64,
                        volumenodes::AbstractString, #"GLL" or "GL"
                        basisnodes::AbstractString, #"GLL" or "GL"
                        fr_c_name::AbstractString, # cDG, cPlus, cHU, cSD, c-, 1000
                        debugmode::Bool
                        ) = new(
                            dim::Int64,
                            n_times_to_solve::Int64,
                            P::Int64,
                            numerical_flux_type::AbstractString,
                            pde_type::AbstractString,
                            usespacetime::Bool,
                            include_source::Bool,
                            alpha_split::Float64,
                            advection_speed::Float64,
                            finaltime::Float64,
                            volumenodes::AbstractString, #"GLL" or "GL"
                            basisnodes::AbstractString, #"GLL" or "GL"
                            fr_c_name::AbstractString, # cDG, cPlus, cHU, cSD, c-, 1000
                            debugmode::Bool
                            )


end

function set_FR_value(param::PhysicsAndFluxParams)
    P = param.P
    fr_c_name = param.fr_c_name

    if cmp(fr_c_name, "cDG") == 0
        param.fluxreconstructionC = 0.0
    elseif cmp(fr_c_name, "cPlus") == 0
        display("WARNING: cPlus not yet set. Returning 0.")
        param.fluxreconstructionC = 0.0
        # set values here
    elseif cmp(fr_c_name, "cHU") == 0
        cp = factorial(2*P)/2^P / factorial(P)^2 # eq. 24 of cicchino 2021 tensor product
        # table 1, cicchino 2021
        param.fluxreconstructionC = (P+1) / (  P* ((2*P+1) * (factorial(P) * cp) ^2)   ) 
    elseif cmp(fr_c_name, "cSD") == 0
        cp = factorial(2*P)/2^P / factorial(P)^2 # eq. 24 of cicchino 2021 tensor product
        # table 1, cicchino 2021
        param.fluxreconstructionC = P / (  (P+1)* ((2*P+1) * (factorial(P) * cp) ^2)   ) 
    elseif cmp(fr_c_name, "1000") == 0
        param.fluxreconstructionC = 1000.0
    elseif cmp(fr_c_name, "c-") == 0
        cp = factorial(2*P)/2^P / factorial(P)^2 # eq. 24 of cicchino 2021 tensor product
        # table 1, cicchino 2021
        param.fluxreconstructionC = -1 / (  ((2*P+1) * (factorial(P) * cp) ^2)   ) 
    end

   
end

function parse_param_Float64(name::String, paramDF)
    return parse(Float64, paramDF[in.(paramDF.name, Ref([name])), "value"][1])
end

function parse_param_Bool(name::String, paramDF)
    return parse(Bool, paramDF[in.(paramDF.name, Ref([name])), "value"][1])
end

function parse_param_Int64(name::String, paramDF)
    return parse(Int64, paramDF[in.(paramDF.name, Ref([name])), "value"][1])
end

function parse_param_String(name::String, paramDF)
    return paramDF[in.(paramDF.name, Ref([name])), "value"][1]
end

function parse_default_parameters()
    #parse default params 
    paramDF = CSV.read("default_parameters.csv", DataFrames.DataFrame)
    display("Default parameter values:")
    display(paramDF)

    dim = parse_param_Int64("dim", paramDF)
    n_times_to_solve = parse_param_Int64("n_times_to_solve", paramDF)
    P = parse_param_Int64("P", paramDF)
    numerical_flux_type = parse_param_String("numerical_flux_type", paramDF)
    pde_type = parse_param_String("pde_type", paramDF)
    usespacetime = parse_param_Bool("usespacetime", paramDF)
    include_source = parse_param_Bool("include_source", paramDF)
    alpha_split = parse_param_Float64("alpha_split", paramDF)
    advection_speed = parse_param_Float64("advection_speed", paramDF)
    finaltime = parse_param_Float64("finaltime", paramDF)
    volumenodes = parse_param_String("volumenodes", paramDF)
    basisnodes = parse_param_String("basisnodes", paramDF)
    fr_c_name = parse_param_String("fr_c_name",paramDF)
    debugmode = parse_param_Bool("debugmode", paramDF)

    param = PhysicsAndFluxParams(
                                 dim,
                                 n_times_to_solve, 
                                 P,  
                                 numerical_flux_type, 
                                 pde_type, 
                                 usespacetime, 
                                 include_source, 
                                 alpha_split, 
                                 advection_speed, 
                                 finaltime, 
                                 volumenodes, 
                                 basisnodes, 
                                 fr_c_name, 
                                 debugmode
                                )
    set_FR_value(param)

    return param
end

function parse_parameters(fname::String)
    default_params = parse_default_parameters()

    if cmp(fname, "default_parameters.csv")!=0 # no need to re-read the file if we are already using default 
        
        newparamDF = CSV.read(fname, DataFrames.DataFrame,types=String)
        display("Custom parameter values, which will override defaults:")
        display(newparamDF)

        if "dim" in newparamDF.name
            default_params.dim = parse_param_Float64("dim", newparamDF)
        end
        if "n_times_to_solve" in newparamDF.name
            default_params.n_times_to_solve = parse_param_Int64("n_times_to_solve", newparamDF)
        end
        if "P" in newparamDF.name
            default_params.P = parse_param_Int64("P", newparamDF)
        end
        if "numerical_flux_type" in newparamDF.name
            default_params.numerical_flux_type = parse_param_String("numerical_flux_type", newparamDF)
        end
        if "pde_type" in newparamDF.name
            default_params.pde_type = parse_param_String("pde_type", newparamDF)
        end
        if "usespacetime" in newparamDF.name
            default_params.usespacetime = parse_param_Bool("usespacetime", newparamDF)
        end
        if "include_source" in newparamDF.name
            default_params.include_source = parse_param_Bool("include_source", newparamDF)
        end
        if "alpha_split" in newparamDF.name
            default_params.alpha_split = parse_param_Float64("alpha_split", newparamDF)
        end
        if "advection_speed" in newparamDF.name
            default_params.advection_speed = parse_param_Float64("advection_speed", newparamDF)
        end
        if "finaltime" in newparamDF.name
            default_params.finaltime = parse_param_Float64("finaltime", newparamDF)
        end
        if "volumenodes" in newparamDF.name
            default_params.volumenodes = parse_param_String("volumenodes", newparamDF)
        end
        if "basisnodes" in newparamDF.name
            default_params.basisnodes = parse_param_String("basis_nodes", newparamDF)
        end
        if "fr_c_name" in newparamDF.name
            default_params.fr_c_name= parse_param_String("fr_c_name", newparamDF)
            set_FR_value(default_params)
        end
        if "debugmode" in newparamDF.name
            default_params.debugmode = parse_param_Bool("debugmode", newparamDF)
        end
    end

    return default_params
end

function transform_physical_to_reference(f_physical, direction, dg::DG)
    return dg.C_m[direction,direction] * f_physical 
end

function calculate_numerical_flux(uM_face,uP_face,n_face, direction,dg::DG, param::PhysicsAndFluxParams)
    f_numerical=zeros(size(uM_face))

    if direction == 2 && param.usespacetime 
        # f_numerical = 0.5 * ( uM_face + uP_face )
        # second direction corresponding to time.
        # only use one-sided information such that the flow of information is from past to future.
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
        return 0.2*sin.(π * (x))
    else
        return sin.(π * (x)) .+ 0.01
    end
end
