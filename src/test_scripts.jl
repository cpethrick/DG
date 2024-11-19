
import DelimitedFiles
include("parameters.jl")

function c_ramp_test(paramfile::String="default_parameters.csv")
    param = parse_parameters(paramfile)
    param.fr_c_name = "user-defined"

    fname = paramfile[1:end-4]*"_result.csv"

    fr_c_ramp_values = exp10.(range(-7.0, 4.0, length=31))
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
