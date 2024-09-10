#==============================================================================
# Parameters for controlling the problem.
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
    spacetime_decouple_slabs::Bool
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
                         spacetime_decouple_slabs::Bool,
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
                                spacetime_decouple_slabs::Bool,
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
    else
        param.fluxreconstructionC = 0.0
        display("Warning! Illegal FR c name!")
    end

    display("FR param is " )
    display(param.fluxreconstructionC)


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
    spacetime_decouple_slabs = parse_param_Bool("spacetime_decouple_slabs", paramDF)
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
                                 spacetime_decouple_slabs,
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
        if "spacetime_decouple_slabs" in newparamDF.name
            default_params.spacetime_decouple_slabs= parse_param_Bool("spacetime_decouple_slabs", newparamDF)
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

