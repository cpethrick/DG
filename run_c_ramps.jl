include("src/main.jl")
# OOA tests
c_ramp_test("c_ramp_tests/lin_adv_P2_GLLGL.csv")
c_ramp_test("c_ramp_tests/lin_adv_P3_GLLGL.csv")
c_ramp_test("c_ramp_tests/lin_adv_P4_GLLGL.csv")
# Stability tests
c_ramp_test("c_ramp_tests/spacetime_burgers_stability_GLLGL_p3.csv", false, true)
c_ramp_test("c_ramp_tests/spacetime_burgers_stability_GLLGL_p4.csv", false, true)
