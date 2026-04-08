
import DelimitedFiles
import PyPlot
using LaTeXStrings
include("parameters.jl")
include("post_processing.jl")


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

function c_ramp_test(paramfile::String="default_parameters.csv", special_c_values=false, stability_ramp=false)
    # Run as: `c_ramp_test("c_ramp/spacetime_linear_advection_P3.csv")`
    #
    #
    #special_c_values = false # run c values cDG, cHu, cSD
    param = parse_parameters(paramfile)
    param.fr_c_name = "user-defined"

    fname = paramfile[1:end-4]*"_result.csv"
    #fr_c_ramp_values = exp10.(range(-5.0, 0.0, length=6))
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

    if stability_ramp == true
        N_small = 8
    end

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
        (err_L2_small, err_Linf_small, entropy_ch_small,cost_tracker) = setup_and_solve(N_small, param.P, param)
        if stability_ramp==false
            (err_L2_big, err_Linf_big, entropy_ch_big, cost_tracker) = setup_and_solve(N_big, param.P, param)
            L2_OOA = log(err_L2_small/err_L2_big) / log(2.0)
        else
            L2_OOA = false
            entropy_ch_big = 0
        end
        f = open(fname, "a")
        DelimitedFiles.writedlm(f, [fr_c_val,L2_OOA,cost_tracker.n_assemble_residual,entropy_ch_big,entropy_ch_small]', ",")
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
    param.volumenodes = "GL"
    param.basisnodes = "GL"
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
    x_quad,y_quad = build_coords_vectors_2D(dg.r_quad, dg.r_quad_y, dg) 
    #x_quad,y_quad = build_coords_vectors(dg.r_quad, dg)  
    PyPlot.plot(x_quad, y_quad, "o", color="#b9b4b3", label=L"$\bar{\xi}_{\mathrm{q}}$, flux nodes", markersize=4)

    x_face = vcat(-1*ones(size(dg.r_quad_y)), ones(size(dg.r_quad_y)), dg.r_quad, dg.r_quad)
    y_face = vcat(dg.r_quad_y, dg.r_quad_y,-1*ones(size(dg.r_quad)), ones(size(dg.r_quad)))
    PyPlot.plot(x_face, y_face, "o", color="#1896cb", label=L"$\bar{\xi}_{\mathrm{f}}$, face nodes", markersize=3)

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


function compare_MoL_spacetime_figure()

    paramfile::AbstractString="default_parameters.csv"
    param = parse_parameters(paramfile)
    param.dim = 1
    param.N_elem_per_dim_coarse=2
    param.P = 3
    param.usespacetime = false
    param.volumenodes = "GLL"
    param.basisnodes = "GLL"
    param.fluxnodes = "GL"
    param.fluxnodes_overintegration = 0
    param.pde_type = "linear_adv_1D"
    param.alpha_split=1
    param.include_source=false
    param.advection_speed=0.3

    dg_MoL = init_DG(param.P, param.dim, 2, 1, [0.0,2], param.volumenodes, param.basisnodes, param.fluxnodes, param.fluxnodes_overintegration, param.fluxreconstructionC, param.usespacetime)

    u0 = calculate_initial_solution(dg_MoL, param)
    u_hat0 = zeros(dg_MoL.N_soln_dof_global)
    u_local_state = zeros(dg_MoL.N_soln)
    for ielem = 1:dg_MoL.N_elem
        for istate = 1:dg_MoL.N_state
            for inode = 1:dg_MoL.N_vol
                u_local_state[inode] = u0[dg_MoL.StIDGIDtoGSID[istate,dg_MoL.EIDLIDtoGID_vol[ielem,inode]]]
            end
            u_hat_local_state = dg_MoL.Pi_soln*u_local_state
            u_hat0[dg_MoL.StIDGIDtoGSID[istate,dg_MoL.EIDLIDtoGID_basis[ielem,:]]] = u_hat_local_state
        end
    end

    CFL = 0.3
    #xmin = minimum(abs.(x[1,:] .- x[2,:]))
    #dt = abs(CFL / a * xmin /2)
    dt = CFL * (dg_MoL.delta_x / dg_MoL.N_soln_per_dim)
    Nsteps::Int64 = ceil(param.finaltime/dt)
    dt = param.finaltime/Nsteps
    if param.debugmode == true
        Nsteps = 1
    end
    display("Beginning time loop")
    (u_hat_MoL,current_time) = physicaltimesolve(u_hat0, dt, Nsteps, dg_MoL, param)
    display("Done time loop")

    (x_overint_1D,x_overint, y_overint,u0_overint_1D, u_calc_final_overint_1D, 
       u_exact_overint_1D, u0_overint,u_calc_final_overint,u_exact_overint) = post_process(u_hat_MoL, current_time, u_hat0, dg_MoL, param, true)

    ########################## plot MoL: grid + initial solution + final solution.

    # Plot settings from python codes
    PyPlot.rc("legend", frameon=false, fontsize="medium")
    #PyPlot.rc('font', family='sans-serif', size=12)
    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", size=12)
    PyPlot.rc("font", family = "sans-serif")

    # define colours
    myred = "#ED1B2F"
    myblue = "#1896cb"
    mygrey = "#b9b4b3"

    PyPlot.figure("Method of Lines - Grid", figsize=(4,1))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks(dg_MoL.VX, minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    PyPlot.axhline(0, 0,2, color="k", linewidth=1.5)
    PyPlot.yticks([])
    PyPlot.plot(dg_MoL.x, dg_MoL.y, "o", color="darkolivegreen", label="volume nodes")
    pltname = string("grid_MoL",".pdf")
    PyPlot.savefig(pltname)

    (fig, axs) = PyPlot.subplots(2,1,figsize=(3,3))
    PyPlot.sca(axs[1])
    ax = axs[1]
    ax.set_xticks(dg_MoL.VX, minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    PyPlot.axhline(0, 0,2, color="k", linewidth=1.5)
    PyPlot.plot(dg_MoL.x, dg_MoL.y, "o", color=myblue, label="volume nodes")
    PyPlot.yticks([])
    PyPlot.plot(x_overint, u_calc_final_overint, color=myred)
    PyPlot.ylabel(L"$t=t_f$")
    ax = axs[2]
    PyPlot.sca(axs[2])
    ax.set_xticks(dg_MoL.VX, minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    PyPlot.axhline(0, 0,2, color="k", linewidth=1.5)
    PyPlot.plot(dg_MoL.x, dg_MoL.y, "o", color=myblue, label="volume nodes")
    PyPlot.yticks([])
    PyPlot.plot(x_overint, u0_overint, color=myred)
    pltname = "soln_MoL.pdf"
    PyPlot.ylabel(L"$t=0$")
    PyPlot.xlabel("Spatial dimension")
    PyPlot.savefig(pltname, bbox_inches="tight")


    ###################################
    param.dim = 2
    param.usespacetime = true
    param.spacetime_solver_type="JFNK"

    dg_ST = init_DG(param.P, param.dim, 2, 1, [0,2.0], param.volumenodes, param.basisnodes, param.fluxnodes, param.fluxnodes_overintegration, param.fluxreconstructionC, param.usespacetime)

    u0 = calculate_initial_solution(dg_ST, param)
    u_hat0 = zeros(dg_ST.N_soln_dof_global)
    u_local_state = zeros(dg_ST.N_soln)
    for ielem = 1:dg_ST.N_elem
        for istate = 1:dg_ST.N_state
            for inode = 1:dg_ST.N_vol
                u_local_state[inode] = u0[dg_ST.StIDGIDtoGSID[istate,dg_ST.EIDLIDtoGID_vol[ielem,inode]]]
            end
            u_hat_local_state = dg_ST.Pi_soln*u_local_state
            u_hat0[dg_ST.StIDGIDtoGSID[istate,dg_ST.EIDLIDtoGID_basis[ielem,:]]] = u_hat_local_state
        end
    end

    cost_tracker = init_CostTracking()
    u_hat_ST = spacetimeimplicitsolve(u_hat0, dg_ST, param, cost_tracker)

    (x_overint_1D,x_overint, y_overint,u0_overint_1D, u_calc_final_overint_1D, 
     u_exact_overint_1D, u0_overint,u_calc_final_overint,u_exact_overint) =   post_process(u_hat_ST, 0.0, u_hat0, dg_ST, param, true)


    ########################## plot ST: grid + initial solution + final solution.

    # Plot settings from python codes
    PyPlot.rc("legend", frameon=false, fontsize="medium")
    #PyPlot.rc('font', family='sans-serif', size=12)
    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", size=12)
    PyPlot.rc("font", family = "sans-serif")


    PyPlot.figure("Spacetime - Grid", figsize=(3,3), dpi=300)
    ax = PyPlot.gca()
    ax.set_xticks(dg_MoL.VX, minor=false)
    ax.set_yticks(dg_MoL.VX, minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.yaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    PyPlot.axhline(0, 0,2, color="k", linewidth=1.5)
    PyPlot.axhline(2, 0,2, color="k", linewidth=1.5)
    PyPlot.axvline(0, 0,2, color="k", linewidth=1.5)
    PyPlot.axvline(2, 0,2, color="k", linewidth=1.5)
    PyPlot.plot(dg_ST.x, dg_ST.y, "o", color=myblue, label="volume nodes")
    PyPlot.xlim([0,2])
    PyPlot.ylim([0,2])
    PyPlot.xlabel("Spatial dimension")
    PyPlot.ylabel("Temporal dimension")
    pltname = string("grid_ST",".pdf")
    PyPlot.savefig(pltname, bbox_inches="tight", dpi=300)

    PyPlot.figure("Spacetime - Solution", figsize=(3,3), dpi=300)
    ax = PyPlot.gca()
    ax.set_xticks(dg_MoL.VX, minor=false)
    ax.set_yticks(dg_MoL.VX, minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.yaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    PyPlot.tricontourf(x_overint, y_overint, u_calc_final_overint, 20)
    PyPlot.xlim([0,2])
    PyPlot.ylim([0,2])
    PyPlot.xlabel("Spatial dimension")
    PyPlot.ylabel("Temporal dimension")
    pltname = "soln_ST.pdf"
    PyPlot.savefig(pltname, bbox_inches="tight", dpi=300)
end
