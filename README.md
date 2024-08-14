# Code Description

The included code implements modal discontinuous Galerkin in `julia` using minimal dependencies. The code is mainly for personal use and therefore is not comprehensively documented. 

# Current features

* Achieves p+1 convergence for linear advection (1D), Burgers' (1D), Burgers' (1D on a 2D grid) when using LxF upwinding.
* Achieves machine precision energy conservation for Burgers' (1D and 2D)
* Can use an arbitrary combination of Gauss-Legendre and Gauss-Lobatto-Legendre nodes for the basis (interpolation) and volume (integration) nodes

# Installing

First, ensure that `julia` is installed:
* Install [juliaup](https://github.com/JuliaLang/juliaup) using `curl -fsSL https://install.julialang.org | sh -s`. This will install the most current version of Julia. The code is known to work in version 1.10.4 of Julia. Continue with the default installation options and restart the terminal to ensure `PATH` is up-to-date.
* Clone this repo to a directory of your choosing.
* In the installation directory, run the command `julia --project=.` to enter the [REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) with the project environment for this code.
* Press `]` to enter package mode. Enter the command `instantiate` to install the dependencies.
* Press backspace to return to julian mode. Test the code by running `include("src/main.jl")`, which will compile and run the most recent commit.
To exit the REPL, use the command `exit()` or press `ctrl+D`.

# Running

Ensure that you are in the julia REPL in julia mode -- `julia>` should appear at the beginning of the line. If you are in the command or package REPL, press backspace to enter julian mode. To launch the REPL, use the command `julia --project=.` . To run the code, use the command `include("src/main.jl")`. This may take some time to compile the first time you run in a particular session.

Currently, the parameters are modified in the `main()` function of `main.jl`. Functional options are clearly indicated. After modifying the code, re-run `include("src/main.jl")`. Julia will only re-compile functions which have been changed.
