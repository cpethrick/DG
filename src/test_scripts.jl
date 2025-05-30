
import DelimitedFiles
import PyPlot
using LaTeXStrings
include("parameters.jl")


function return_dg_object(N_elem_per_dim, paramfile::String="default_parameters.csv")
    param = parse_parameters(paramfile)

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
    return init_DG(param.P, param.dim, N_elem_per_dim, N_state, [x_Llim,x_Rlim], param.volumenodes, param.basisnodes, param.fluxnodes, param.fluxnodes_overintegration, param.fluxreconstructionC, param.usespacetime)
end

function c_ramp_test(paramfile::String="default_parameters.csv")
    # Run as: `c_ramp_test("c_ramp/spacetime_linear_advection_P3.csv")`
    #
    #
    special_c_values = true # run c values cDG, cHu, cSD
    param = parse_parameters(paramfile)
    param.fr_c_name = "user-defined"

    fname = paramfile[1:end-4]*"_result.csv"
    fr_c_ramp_values = exp10.(range(-7.0, 4.0, length=31))
    fr_c_names = []
    if special_c_values
        fr_c_ramp_values = [0, 0, 0]
        fr_c_names = ["cDG", "cSD", "cHu"]
        fname = paramfile[1:end-4]*"_result_specialCvalues.csv"
    end
   
    #can append special values here (cHU, cSD)

    N_small = 16
    N_big = N_small * 2

    f = open(fname, "w")
    DelimitedFiles.writedlm(f, ["c_value" "OOA" "number_iterations"], ",")
    close(f)
    for i in 1:length(fr_c_ramp_values)
        fr_c_val = fr_c_ramp_values[i]
        param.fr_c_userdefined = fr_c_val
        if special_c_values
            param.fr_c_name = fr_c_names[i]
        end
        set_FR_value(param)
        fr_c_val = param.fluxreconstructionC
        (err_L2_small, err_Linf_small, entropy_ch_small,_) = setup_and_solve(N_small, param.P, param)
        (err_L2_big, err_Linf_big, entropy_ch_big, cost_tracker) = setup_and_solve(N_big, param.P, param)
        L2_OOA = log(err_L2_small/err_L2_big) / log(2.0)
        f = open(fname, "a")
        DelimitedFiles.writedlm(f, [fr_c_val,L2_OOA,cost_tracker.n_assemble_residual]', ",")
        close(f)
    end


end

function reference_element_figure()

    # Function to plot a single labelled reference element.
   
    # Define a param object from the default settings and manually override some values

    paramfile::AbstractString="default_parameters.csv"
    param = parse_parameters(paramfile)
    param.dim = 2
    param.P = 3
    param.usespacetime = true
    param.volumenodes = "GLL"
    param.basisnodes = "GLL"
    param.fluxnodes = "GL"
    param.fluxnodes_overintegration = 0

    dg = init_DG(param.P, param.dim, 1, 1, [-1.0,1.0], param.volumenodes, param.basisnodes, param.fluxnodes, param.fluxnodes_overintegration, param.fluxreconstructionC, param.usespacetime)


    # Plot settings from python codes
    PyPlot.rc("legend", frameon=false, fontsize="medium")
    #PyPlot.rc('font', family='sans-serif', size=12)
    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", size=8)
    PyPlot.rc("font", family = "serif")

    PyPlot.figure("Reference Element", figsize=(4,2.5))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks([-1.0, 0.0, 1.0], minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.set_yticks([-1.0, 0.0, 1.0], minor=false)
    ax.yaxis.grid(true, which="major", color="k")
    #ax.set_axisbelow(false)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.spines["left"].set_visible(false)
    PyPlot.ylabel(L"$\tau$, temporal coordinate")
    PyPlot.xlabel(L"$\xi$, spatial coordinate")
    PyPlot.plot(dg.x, dg.y, "o", color="#ED1B2F", label=L"$\bar{\xi}_{\mathrm{soln}}$, solution nodes", markersize=5)
    x_flux,y_flux = build_coords_vectors(dg.r_flux, dg)  
    PyPlot.plot(x_flux, y_flux, "o", color="#b9b4b3", label=L"$\bar{\xi}_{\mathrm{flux}}$, flux nodes", markersize=4)

    x_face = vcat(-1*ones(size(dg.r_flux)), ones(size(dg.r_flux)), dg.r_flux, dg.r_flux)
    y_face = vcat(dg.r_flux, dg.r_flux,-1*ones(size(dg.r_flux)), ones(size(dg.r_flux)))
    PyPlot.plot(x_face, y_face, "o", color="#1896cb", label=L"$\bar{\xi}_{\mathrm{face}}$, face nodes", markersize=3)

    box = ax.get_position()
    #shrink plot
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #place plot outside box
    PyPlot.legend(loc = "center left", bbox_to_anchor=(1,0.5))


    PyPlot.plot([-1,1,1,-1,-1], [-1, -1, 1, 1, -1], "k", zorder = 1)


    ax.text(0.2,-1.2,L"\textbf{Face 3}", horizontalalignment="center")
    ax.text(-1.1,0.1,L"\textbf{Face 1}", horizontalalignment="center", rotation="vertical")
    ax.text(0.2,1.1,L"\textbf{Face 4}", horizontalalignment="center")
    ax.text(1.15,0.1,L"\textbf{Face 2}", horizontalalignment="center", rotation="vertical")

    PyPlot.savefig("reference_element.pdf", bbox_inches="tight")



end
