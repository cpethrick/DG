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
    domain_size::Float64
    numerical_flux_type::AbstractString
    pde_type::AbstractString
    usespacetime::Bool
    spacetime_decouple_slabs::Bool
    spacetime_solver_type::AbstractString #pseudotime or JFNK
    spacetime_JFNK_solver_log::Bool #to control printing of the solver log to a file
    include_source::Bool
    alpha_split::Float64
    use_skew_symmetric_stiffness_operator::Bool
    strong_in_time::Bool # Flag for doing NSFR in space and DG (non-split) in time
    advection_speed::Float64
    finaltime::Float64
    volumenodes::String #"GLL" or "GL"
    basisnodes::String #"GLL" or "GL"
    fluxnodes::String # "GLL" or "GL"
    fluxnodes_overintegration::Int64
    fr_c_name::String # cDG, cPlus, cHU, cSD, c-, 1000, user-defined, case-insensitive
    fr_c_userdefined::Float64
    do_conservation_check::Bool # Controls how well-converged the solution is
    debugmode::Bool
    convergence_table_name::AbstractString # none or descriptive name
    read_soln_from_file::Bool

    # dependant params: set based on above required params.
    #set based on value of fr_c_name
    fluxreconstructionC::Float64


    # incomplete initialization: leave dependant variables uninitialized.
    PhysicsAndFluxParams(
                         dim::Int64,
                         n_times_to_solve::Int64,
                         P::Int64,
                         domain_size::Float64,
                         numerical_flux_type::AbstractString,
                         pde_type::AbstractString,
                         usespacetime::Bool,
                         spacetime_decouple_slabs::Bool,
                         spacetime_solver_type::AbstractString, 
                         spacetime_JFNK_solver_log::Bool,
                         include_source::Bool,
                         alpha_split::Float64,
                         use_skew_symmetric_stiffness_operator::Bool,
                         strong_in_time::Bool,
                         advection_speed::Float64,
                         finaltime::Float64,
                         volumenodes::AbstractString, #"GLL" or "GL"
                         basisnodes::AbstractString, #"GLL" or "GL"
                         fluxnodes::AbstractString, #"GLL" or "GL"
                         fluxnodes_overintegration::Int64,
                         fr_c_name::AbstractString, # cDG, cPlus, cHU, cSD, c-, 1000, user-defined
                         fr_c_userdefined::Float64,
                         do_conservation_check::Bool,
                         debugmode::Bool,
                         convergence_table_name::AbstractString,
                         read_soln_from_file::Bool
                        ) = new(
                                dim::Int64,
                                n_times_to_solve::Int64,
                                P::Int64,
                                domain_size::Float64,
                                numerical_flux_type::AbstractString,
                                pde_type::AbstractString,
                                usespacetime::Bool,
                                spacetime_decouple_slabs::Bool,
                                spacetime_solver_type::AbstractString,
                                spacetime_JFNK_solver_log::Bool,
                                include_source::Bool,
                                alpha_split::Float64,
                                use_skew_symmetric_stiffness_operator::Bool,
                                strong_in_time::Bool,
                                advection_speed::Float64,
                                finaltime::Float64,
                                volumenodes::AbstractString, #"GLL" or "GL"
                                basisnodes::AbstractString, #"GLL" or "GL"
                                fluxnodes::AbstractString, #"GLL" or "GL"
                                fluxnodes_overintegration::Int64,
                                fr_c_name::AbstractString, # cDG, cPlus, cHU, cSD, c-, 1000, user-defined
                                fr_c_userdefined::Float64,
                                do_conservation_check::Bool,
                                debugmode::Bool,
                                convergence_table_name::AbstractString,
                                read_soln_from_file::Bool
                               )


end

function set_FR_value(param::PhysicsAndFluxParams)
    P = param.P
    fr_c_name = lowercase(param.fr_c_name)

    if cmp(fr_c_name, "cdg") == 0
        param.fluxreconstructionC = 0.0
    elseif cmp(fr_c_name, "cplus") == 0
        display("WARNING: cPlus not yet set. Returning 0.")
        param.fluxreconstructionC = 0.0
        # set values here
    elseif cmp(fr_c_name, "chu") == 0
        cp = factorial(2*P)/2^P / factorial(P)^2 # eq. 24 of cicchino 2021 tensor product
        # table 1, cicchino 2021
        param.fluxreconstructionC = (P+1) / (  P* ((2*P+1) * (factorial(P) * cp) ^2)   ) 
    elseif cmp(fr_c_name, "csd") == 0
        cp = factorial(2*P)/2^P / factorial(P)^2 # eq. 24 of cicchino 2021 tensor product
        # table 1, cicchino 2021
        param.fluxreconstructionC = P / (  (P+1)* ((2*P+1) * (factorial(P) * cp) ^2)   ) 
    elseif cmp(fr_c_name, "1000") == 0
        param.fluxreconstructionC =1.0E3
    elseif cmp(fr_c_name, "c-") == 0
        cp = factorial(2*P)/2^P / factorial(P)^2 # eq. 24 of cicchino 2021 tensor product
        # table 1, cicchino 2021
        param.fluxreconstructionC = -1 / (  ((2*P+1) * (factorial(P) * cp) ^2)   ) 
    elseif cmp(fr_c_name, "user-defined") == 0
        param.fluxreconstructionC = param.fr_c_userdefined
    else
        param.fluxreconstructionC = 0.0
        display("Warning! Illegal FR c name!")
    end

    display("Value of C is:")
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
    domain_size = parse_param_Float64("domain_size", paramDF)
    numerical_flux_type = parse_param_String("numerical_flux_type", paramDF)
    pde_type = parse_param_String("pde_type", paramDF)
    usespacetime = parse_param_Bool("usespacetime", paramDF)
    spacetime_decouple_slabs = parse_param_Bool("spacetime_decouple_slabs", paramDF)
    spacetime_solver_type = parse_param_String("spacetime_solver_type", paramDF)
    spacetime_JFNK_solver_log = parse_param_Bool("spacetime_JFNK_solver_log", paramDF)
    include_source = parse_param_Bool("include_source", paramDF)
    alpha_split = parse_param_Float64("alpha_split", paramDF)
    use_skew_symmetric_stiffness_operator = parse_param_Bool("use_skew_symmetric_stiffness_operator", paramDF)
    strong_in_time= parse_param_Bool("strong_in_time", paramDF)
    advection_speed = parse_param_Float64("advection_speed", paramDF)
    finaltime = parse_param_Float64("finaltime", paramDF)
    volumenodes = parse_param_String("volumenodes", paramDF)
    basisnodes = parse_param_String("basisnodes", paramDF)
    fluxnodes = parse_param_String("fluxnodes", paramDF)
    fluxnodes_overintegration = parse_param_Int64("fluxnodes_overintegration", paramDF)
    fr_c_name = parse_param_String("fr_c_name",paramDF)
    fr_c_userdefined = parse_param_Float64("fr_c_userdefined",paramDF)
    do_conservation_check = parse_param_Bool("do_conservation_check",paramDF)
    debugmode = parse_param_Bool("debugmode", paramDF)
    convergence_table_name = parse_param_String("convergence_table_name",paramDF)
    read_soln_from_file = parse_param_Bool("read_soln_from_file", paramDF)

    param = PhysicsAndFluxParams(
                                 dim,
                                 n_times_to_solve, 
                                 P,
                                 domain_size,
                                 numerical_flux_type, 
                                 pde_type, 
                                 usespacetime, 
                                 spacetime_decouple_slabs,
                                 spacetime_solver_type,
                                 spacetime_JFNK_solver_log,
                                 include_source, 
                                 alpha_split, 
                                 use_skew_symmetric_stiffness_operator,
                                 strong_in_time,
                                 advection_speed, 
                                 finaltime, 
                                 volumenodes, 
                                 basisnodes, 
                                 fluxnodes,
                                 fluxnodes_overintegration,
                                 fr_c_name, 
                                 fr_c_userdefined,
                                 do_conservation_check,
                                 debugmode,
                                 convergence_table_name,
                                 read_soln_from_file
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
        if "domain_size" in newparamDF.name
            default_params.domain_size = parse_param_Float64("domain_size", newparamDF)
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
        if "spacetime_solver_type" in newparamDF.name
            default_params.spacetime_solver_type = parse_param_String("spacetime_solver_type", newparamDF)
        end
        if "spacetime_JFNK_solver_log" in newparamDF.name
            default_params.spacetime_JFNK_solver_log = parse_param_Bool("spacetime_JFNK_solver_log", newparamDF)
        end
        if "include_source" in newparamDF.name
            default_params.include_source = parse_param_Bool("include_source", newparamDF)
        end
        if "alpha_split" in newparamDF.name
            default_params.alpha_split = parse_param_Float64("alpha_split", newparamDF)
        end
        if "use_skew_symmetric_stiffness_operator" in newparamDF.name
            default_params.use_skew_symmetric_stiffness_operator = parse_param_Bool("use_skew_symmetric_stiffness_operator", newparamDF)
        end
        if "strong_in_time" in newparamDF.name
            default_params.strong_in_time= parse_param_Bool("strong_in_time", newparamDF)
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
            default_params.basisnodes = parse_param_String("basisnodes", newparamDF)
        end
        if "fluxnodes" in newparamDF.name
            default_params.fluxnodes = parse_param_String("fluxnodes", newparamDF)
        end
        if "fluxnodes_overintegration" in newparamDF.name
            default_params.fluxnodes_overintegration = parse_param_Int64("fluxnodes_overintegration", newparamDF)
        end
        if "fr_c_userdefined" in newparamDF.name
            #need to define fr_c_userdefined BEFORE fr_c_name
            default_params.fr_c_userdefined= parse_param_Float64("fr_c_userdefined", newparamDF)
        end
        if "fr_c_name" in newparamDF.name
            default_params.fr_c_name= parse_param_String("fr_c_name", newparamDF)
            set_FR_value(default_params)
        end
        if "do_conservation_check" in newparamDF.name
            default_params.do_conservation_check = parse_param_Bool("do_conservation_check", newparamDF)
        end
        if "debugmode" in newparamDF.name
            default_params.debugmode = parse_param_Bool("debugmode", newparamDF)
        end
        if "convergence_table_name" in newparamDF.name
            default_params.convergence_table_name = parse_param_String("convergence_table_name", newparamDF)
        end
        if "read_soln_from_file" in newparamDF.name
            default_params.read_soln_from_file = parse_param_Bool("read_soln_from_file", newparamDF)
        end
    end

    display_param_warnings(default_params)

    return default_params
end

function display_param_warnings(param::PhysicsAndFluxParams)
    # Function to catch known issues and warn the user.
    #

    if cmp(param.volumenodes, param.basisnodes) !=0
        display("****WARNING: order known to drop to 1 if volume and basis nodes not matching!****")
    end

    if param.fluxnodes_overintegration != 0
        display("****WARNING: conservation not holding for overintegrated nodes.****")
        # Likely a simple bug, but haven't yet looked into it. Recommend just using mis-
        # matched soln and flux nodes for now.
    end

    if param.debugmode
        display("****WARNING: Solving in debug mode. This will NOT result in a converged solution.****")
    end





end
