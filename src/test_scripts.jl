
import DelimitedFiles
import PyPlot
include("parameters.jl")

function c_ramp_test(paramfile::String="default_parameters.csv")
    param = parse_parameters(paramfile)
    param.fr_c_name = "user-defined"

    fname = paramfile[1:end-4]*"_result.csv"

    fr_c_ramp_values = exp10.(range(-7.0, 4.0, length=21))
    #can append special values here (cHU, cSD)

    N_small = 16
    N_big = N_small * 2

    f = open(fname, "w")
    DelimitedFiles.writedlm(f, ["c_value" "OOA"], ",")
    close(f)
    for fr_c_val in fr_c_ramp_values
        param.fr_c_userdefined = fr_c_val
        set_FR_value(param)
        (err_L2_small, err_Linf_small, entropy_ch_small) = setup_and_solve(N_small, param.P, param)
        (err_L2_big, err_Linf_big, entropy_ch_big) = setup_and_solve(N_big, param.P, param)
        L2_OOA = log(err_L2_small/err_L2_big) / log(2.0)
        f = open(fname, "a")
        DelimitedFiles.writedlm(f, [fr_c_val,L2_OOA]', ",")
        close(f)
    end


end

function reference_element_figure()

    # Function to plot a single labelled reference element.
   
    # Define a param object from the default settings and manually override some values

    paramfile::AbstractString="default_parameters.csv"
    param = parse_parameters(paramfile)
    param.volumenodes = "GL"

    dg = init_DG(param.P, param.dim, 1, N_state, [-1.0,1.0], param.volumenodes, param.basisnodes, param.fluxreconstructionC, param.usespacetime)

    PyPlot.figure("Reference Element", figsize=(6,6))
    PyPlot.clf()
    ax = PyPlot.gca()
    ax.set_xticks([-1.0, 0.0, 1.0], minor=false)
    ax.xaxis.grid(true, which="major", color="k")
    ax.set_axisbelow(false)
    ax.set_yticks([-1.0, 0.0, 1.0], minor=false)
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

end
