
mutable struct LocalElement
    # Contains DG information which is local to a given element type.
    # Allows for different poly-degrees or bases to be used throughout the domain.
    
    P::Int #polynomial degree
    y_dir_overint::Int #NEW # Difference in polydegree in the second dimension
    

    N_soln_per_dim::Int # Number of points per direction per cell; assumes x-direction
    N_soln_y::Int # number of points in the y direction per cell
    N_quad_per_dim::Int # Number of points per direction per cell; assumes y-direction
    N_quad_y::Int # Number of points in y direction per cell
    N_quad::Int # Total number of points per cell = N_quad^dim 
    N_quad_dof::Int # Total number of DOFs per cell = N_quad*N_state #UNSURE IF THIS IS USED. 
    N_face::Int # points on each face parallel to x-axis. I assume that the face nodes are 1D flux nodes.
    N_face_y::Int # points on each face parallel to y-axis. I assume that the face nodes are 1D flux nodes.
    
    ## Unsure whether the next two would be local or global...
    EIDLIDtoGID_basis::AbstractMatrix{Int} # Index of first dimension is element ID, index of second
                             # dimension is element ID. values are global ID.
    EIDLIDtoGID_soln::AbstractMatrix{Int} # Index of first dimension is element ID, index of second
    StIDLIDtoLSID::AbstractMatrix{Int} # StID is the state ID, 1:Nstate
                                       # LID is the ID of the node
                                       # LSID ("local storage") indicates the index in the storage vector
    
    # 1D quadratures and weights in x-direction
    r_soln::Vector{Float64}
    w_soln::Vector{Float64}
    r_basis::Vector{Float64}
    w_basis::Vector{Float64}
    r_quad::Vector{Float64} #face nodes are 1D flux nodes.
    w_quad::Vector{Float64}
    # 1D quadratures and weights in y-direction
    r_soln_y::Vector{Float64}
    w_soln_y::Vector{Float64}
    r_basis_y::Vector{Float64}
    w_basis_y::Vector{Float64}
    r_quad_y::Vector{Float64} #face nodes are 1D flux nodes.
    w_quad_y::Vector{Float64}
    #chi are basis functions for the solution.
    chi_soln::AbstractMatrix{Float64}
    chi_quad::AbstractMatrix{Float64}
    chi_face::Dict{Int64,AbstractArray{Float64}}
    chi_vf::AbstractMatrix{Float64} # stacked matrices for skew-symmetric form
    #phi are basis functions for the flux.
    phi_quad::AbstractMatrix{Float64}
    phi_face::Dict{Int64,AbstractArray{Float64}}
    d_phi_quad_d_xi::AbstractMatrix{Float64}
    d_phi_quad_d_eta::AbstractMatrix{Float64}
    C_m::AbstractMatrix{Float64}
    
    W_soln::AbstractMatrix{Float64}
    W_face::Dict{Int64,AbstractMatrix{Float64}}
    W_quad::AbstractMatrix{Float64}
    J_soln::Float64
    J_face::Float64
    M::AbstractMatrix{Float64}
    MpK::AbstractMatrix{Float64}
    M_inv::AbstractMatrix{Float64}
    MpK_inv::AbstractMatrix{Float64}
    S_xi::AbstractMatrix{Float64}
    S_eta::AbstractMatrix{Float64}
    S_noncons_xi::AbstractMatrix{Float64}
    S_noncons_eta::AbstractMatrix{Float64}
    Pi_soln::AbstractMatrix{Float64}
    Pi_quad::AbstractMatrix{Float64}
    K::AbstractMatrix{Float64}
    QtildemQtildeT::AbstractArray{Float64}

    #The following are not currently used in the solver, but are included to 
    #have consistent notation with the paper.
    L_xi1::AbstractMatrix{Float64}
    L_xi2::AbstractMatrix{Float64}
    L_tau3::AbstractMatrix{Float64}
    L_tau4::AbstractMatrix{Float64}
    D_xi::AbstractMatrix{Float64}
    D_tau::AbstractMatrix{Float64}
    
    #Incomplete initializer - only assign Category 1 variables.
    LocalElement(P::Int,
                 y_dir_overint::Int) = new(P::Int,
                                           y_dir_overint::Int)
end

function init_LocalElement(P::Int, dim::Int, N_state::Int,
        solnnodes::String, basisnodes::String, quadnodes::String, quadnodes_overintegration::Int, fluxreconstructionC::Float64,
        usespacetime::Bool, y_dir_overintegration::Int)


    le = LocalElem(P,
                   y_dir_overint)

    # Flag to set reference cell from 0 to 1, matching PHiLiP.
    reference_cell_01 = false
    if reference_cell_01 
        display("WARNING!! 2D will break because y-dim assumes (-1,1) reference cell!!")
    end

    # Solution nodes (integration nodes)
    if cmp(solnnodes, "GLL") == 0 
        display("GLL Volume nodes.")
        le.r_soln,le.w_soln = FastGaussQuadrature.gausslobatto(le.N_soln_per_dim)
        le.r_soln_y,le.w_soln_y = FastGaussQuadrature.gausslobatto(le.N_soln_y)
    elseif cmp(solnnodes, "GL") == 0
        display("GL Volume nodes.")
        le.r_soln,le.w_soln = FastGaussQuadrature.gaussjacobi(le.N_soln_per_dim, 0.0,0.0)
        le.r_soln_y,le.w_soln_y = FastGaussQuadrature.gaussjacobi(le.N_soln_y, 0.0,0.0)
    else
        display("Illegal soln node choice!")
    end
    if reference_cell_01
        le.r_soln= le.r_soln * 0.5 .+ 0.5 # for changing ref element to match PHiLiP for debugging purposes
        le.w_soln /= 2.0
    end

    # Basis function nodes (shape functions, interpolation nodes)
    if cmp(basisnodes, "GLL") == 0 
        display("GLL basis nodes.")
        le.r_basis,le.w_basis=FastGaussQuadrature.gausslobatto(le.N_soln_per_dim)
        le.r_basis_y,le.w_basis_y=FastGaussQuadrature.gausslobatto(le.N_soln_y)
    elseif cmp(basisnodes, "GL") == 0
        display("GL basis nodes.")
        le.r_basis,le.w_basis=FastGaussQuadrature.gaussjacobi(le.N_soln_per_dim,0.0,0.0)
        le.r_basis_y,le.w_basis_y=FastGaussQuadrature.gaussjacobi(le.N_soln_y,0.0,0.0)
    else
        display("Illegal basis node choice!")
    end
    if reference_cell_01
        le.r_basis = le.r_basis * 0.5 .+ 0.5
        le.w_basis /= 2.0
    end
    
    # Flux nodes (shape functions, interpolation nodes)
    # Assume collocated flux basis nodes.
    if cmp(quadnodes, "GLL") == 0 
        display("GLL flux nodes.")
        le.r_quad,le.w_quad=FastGaussQuadrature.gausslobatto(le.N_quad_per_dim)
        le.r_quad_y,le.w_quad_y=FastGaussQuadrature.gausslobatto(le.N_quad_y)
    elseif cmp(quadnodes, "GL") == 0
        display("GL flux nodes.")
        le.r_quad,le.w_quad=FastGaussQuadrature.gaussjacobi(le.N_quad_per_dim,0.0,0.0)
        le.r_quad_y,le.w_quad_y=FastGaussQuadrature.gaussjacobi(le.N_quad_y,0.0,0.0)
    else
        display("Illegal flux node choice!")
    end
    if quadnodes_overintegration>0
        display("Overintegrating the flux by")
        display(quadnodes_overintegration)
    end
    if reference_cell_01
        le.r_quad = le.r_quad * 0.5 .+ 0.5
        le.w_quad /= 2.0
    end
    

    # Define Vandermonde matrices
    if dim == 1
        le.chi_soln = vandermonde1D(le.r_soln,le.r_basis)
        le.phi_quad = vandermonde1D(le.r_quad,le.r_quad)
        le.d_phi_quad_d_xi = gradvandermonde1D(le.r_quad,le.r_quad)
        d_chi_quad_d_xi = gradvandermonde1D(le.r_quad,le.r_basis)
        #reference coordinates of L and R faces
        r_f_L::Float64 = 0
        if reference_cell_01
            r_f_L= 0
        else
            r_f_L= -1
        end
        r_f_R::Float64 = 1
        le.chi_face = assembleFaceVandermonde1D(r_f_L,r_f_R,le.r_basis)
        le.phi_face = assembleFaceVandermonde1D(r_f_L,r_f_R,le.r_quad)
        le.chi_quad = vandermonde1D(le.r_quad,le.r_basis)

        le.W_soln = LinearAlgebra.diagm(le.w_soln) # diagonal matrix holding quadrature weights
        le.W_quad = LinearAlgebra.diagm(le.w_quad) # diagonal matrix holding quadrature weights
        le.W_face = Dict{Int64, AbstractArray{Float64}}()
        le.W_face[1] = reshape([1.0], 1, 1) #1x1 matrix for generality with higher dim
        le.W_face[2] = reshape([1.0], 1, 1) #1x1 matrix for generality with higher dim
        le.C_m = reshape([1.0], 1, 1) #1x1 matrix for generality with higher dim
    elseif dim == 2
        le.chi_soln = vandermonde2D(le.r_soln,le.r_basis,le.r_soln_y,le.r_basis_y, le)
        le.chi_quad = vandermonde2D(le.r_quad,le.r_basis,le.r_quad_y,le.r_basis_y, le)
        le.phi_quad = vandermonde2D(le.r_quad,le.r_quad, le.r_quad_y,le.r_quad_y,le)
        le.d_phi_quad_d_xi = gradvandermonde2D(1, le.r_quad,le.r_quad, le.r_quad_y,le.r_quad_y,le)
        le.d_phi_quad_d_eta = gradvandermonde2D(2, le.r_quad,le.r_quad,le.r_quad_y,le.r_quad_y,le)
        d_chi_quad_d_xi = gradvandermonde2D(1, le.r_quad,le.r_basis,le.r_quad_y,le.r_basis_y, le)
        d_chi_quad_d_eta = gradvandermonde2D(2, le.r_quad,le.r_basis,le.r_quad_y,le.r_basis_y, le)
        if reference_cell_01
            display("Note that -1 1 cell is hardcoded in assembleFaceVandermonde2D")
        end
        le.chi_face = assembleFaceVandermonde2D(le.r_basis, le.r_quad, le.r_basis_y, le.r_quad_y,le) #face nodes are 1D flux nodes
        le.phi_face = assembleFaceVandermonde2D(le.r_quad, le.r_quad, le.r_quad_y, le.r_quad_y,le) #face nodes are 1D flux nodes
        ##### Check here if stuff doesn't work
        le.W_soln = LinearAlgebra.diagm(vec(le.w_soln*le.w_soln_y'))
        le.W_quad = LinearAlgebra.diagm(vec(le.w_quad*le.w_quad_y'))
        le.W_face = Dict{Int64, AbstractArray{Float64}}()
        le.W_face[1] = LinearAlgebra.diagm(le.w_quad_y)
        le.W_face[2] = LinearAlgebra.diagm(le.w_quad_y)
        le.W_face[3] = LinearAlgebra.diagm(le.w_quad)
        le.W_face[4] = LinearAlgebra.diagm(le.w_quad)
        le.J_face = jacobian ^ (1/dim) # 1D jacobian on the face of the element.
        le.C_m = le.delta_x/cell_length * [1 0; 0 1]  # Assuming a cartesian element
    end

    # for skew-symmetric stiffness operator form
    le.chi_vf = le.chi_quad'
    for iface=1:le.N_faces
        le.chi_vf = [le.chi_vf le.chi_face[iface]' ]
    end
    
    # Mass and stiffness matrices as defined Eq. 9.5 Cicchino 2022
    # All defined on a single element.
    le.M = le.J_soln * le.chi_quad' * le.W_quad * le.chi_quad # J_soln is the same as J_quad (assumin N_overint=0)
    le.S_xi = le.chi_quad' * le.W_quad * le.d_phi_quad_d_xi
    le.S_noncons_xi = le.W_quad * d_chi_quad_d_xi
    if dim==2
        le.S_eta = le.chi_quad' * le.W_quad * le.d_phi_quad_d_eta
        le.S_noncons_eta = le.W_quad * d_chi_quad_d_eta
    end
    M_nojac_soln = le.chi_soln' * le.W_soln * le.chi_soln
    le.Pi_soln = inv(M_nojac_soln)*le.chi_soln'*le.W_soln
    le.K = zeros(size(le.M))
    M_nojac_quad = le.phi_quad' * le.W_quad * le.phi_quad
    le.Pi_quad = inv(M_nojac_quad)*le.phi_quad'*le.W_quad
    if dim==1 && fluxreconstructionC != 0.0
        M_nojac_quad = le.chi_quad' * le.W_quad * le.chi_quad # redefine to use soln basis
        d_chi_quad_d_xi = gradvandermonde1D(le.r_quad,le.r_basis)
        D_xi = inv(M_nojac_quad) * le.chi_quad' * le.W_quad * d_chi_quad_d_xi
        le.K = fluxreconstructionC * ( (D_xi)^P )' * le.M * ( (D_xi)^P ) ## Verified for 1D.
    elseif dim==2 && fluxreconstructionC != 0.0
        #==== This version uses tensor products
        le.K = 0*le.M #initialize with size of M

        chi_v_1D = vandermonde1D(le.r_soln,le.r_basis)
        d_chi_v_d_xi_1D = gradvandermonde1D(le.r_soln,le.r_basis)
        W_1D = le.W_f
        J_1D = le.J_f
        M_1D = chi_v_1D' * W_1D * J_1D * chi_v_1D
        display(M_1D) # ok
        D_xi_1D_P = (inv(chi_v_1D' * W_1D * chi_v_1D) * (chi_v_1D' * W_1D * d_chi_v_d_xi_1D))^P
        display(D_xi_1D_P) # NOT okay.

        K_1D = fluxreconstructionC * (D_xi_1D_P)' * M_1D * D_xi_1D_P
        
        FR1 = tensor_product_2D(K_1D, M_1D)
        FR2 = tensor_product_2D(M_1D, K_1D)
        FRcross = tensor_product_2D(K_1D, K_1D)

        le.K = FR1 + FR2 + FRcross


        Following code calculates per Cicchino 2022 curvilinear eq. 28-29. Does the same thing as above code.
        ==# 
        M_nojac_quad = le.chi_quad' * le.W_quad * le.chi_quad # redefine to use soln basis
        d_chi_quad_d_xi = gradvandermonde2D(1, le.r_quad, le.r_basis, le)
        d_chi_quad_d_eta = gradvandermonde2D(2, le.r_quad, le.r_basis, le)
        D_xi = inv(M_nojac_quad)*le.chi_quad' * le.W_quad * d_chi_quad_d_xi
        D_eta = inv(M_nojac_quad)*le.chi_quad' * le.W_quad * d_chi_quad_d_eta
        le.K = 0*le.M #initialize with size of M
        if usespacetime
            # K only in space
            le.K = fluxreconstructionC * (D_xi^P)' * le.M * D_xi^P
        else
            # K in space and in time
            for s = [0,P]
                for v = [0,P]
                    if s+v >= P
                        c_sv = fluxreconstructionC ^ ((s/P) + (v/P))
                        le.K += c_sv * (D_xi^s * D_eta^v)' * le.M * (D_xi^s * D_eta^v)
                    end
                end
            end
        end
    end
    display("FR K")
    display(le.K) # Verified against PHiLiP for 1D and 2D using C from PHiLiP
    display("Mass")
    display(le.M) # Verified against PHiLiP for 1D and 2D using C from PHiLiP


    le.M_inv = inv(le.M) # unmodified mass matrix.
    le.MpK = le.M + le.K # Here, modified mass matrix.
    display("Adjusted Mass matrix")
    display(le.MpK)
    le.MpK_inv = inv(le.MpK)


    
    if dim==1
        N_face_all = le.N_faces*le.N_face
    elseif dim==2
        N_face_all = 2*le.N_face + 2*le.N_face_y
    end
    Q_dimension = le.N_quad+N_face_all

    le.QtildemQtildeT = zeros(Q_dimension,Q_dimension,dim) 
    # volume quadrature
    le.QtildemQtildeT[1:le.N_quad, 1:le.N_quad,1] .= le.W_quad * le.d_phi_quad_d_xi- le.d_phi_quad_d_xi' * le.W_quad
    if dim==2
        le.QtildemQtildeT[1:le.N_quad, 1:le.N_quad,2] .= le.W_quad * le.d_phi_quad_d_eta- le.d_phi_quad_d_eta' * le.W_quad
    end

    # face - only assemble top-right matrix
    if dim==1
         for iface = 1:le.N_faces
            le.QtildemQtildeT[1:le.N_quad,(le.N_quad+1+le.N_face*(iface-1)):(le.N_quad+le.N_face*iface),1] .+= le.phi_face[iface]' * le.W_face[iface] * le.LFIDtoNormal[iface,1] # 1st direction of normal
        end
    elseif dim==2
        N_face_count = [le.N_face_y, le.N_face_y, le.N_face, le.N_face]
        starting_ind = le.N_quad+1
        for iface = 1:le.N_faces
            ending_ind = starting_ind + N_face_count[iface]-1
            le.QtildemQtildeT[1:le.N_quad,
                              starting_ind:ending_ind,
                              1] .+= le.phi_face[iface]' * le.W_face[iface] * le.LFIDtoNormal[iface,1] # 1st direction of normal
            le.QtildemQtildeT[1:le.N_quad,
                              starting_ind:ending_ind,
                              2] .+= le.phi_face[iface]' * le.W_face[iface] * le.LFIDtoNormal[iface,2] # 2nd direction of normal
            starting_ind += N_face_count[iface]
        end
    end
    # then assign skew-symmetric matrix
    for idim = 1:dim
        le.QtildemQtildeT[le.N_quad+1:end, 1:le.N_quad, idim] .=  -1.0 * le.QtildemQtildeT[1:le.N_quad,le.N_quad+1:end,idim]'
    end
    display(le.QtildemQtildeT)

    #display("Skew-symmetric stiffness operator:")
    #display(le.QtildemQtildeT)
    #
    #
    #The following are not used in calculation, but are constructed consistent with defns in the paper

    if dim==2 && usespacetime
        le.L_tau3 = le.M_inv * le.chi_face[3]' * le.W_face[3] * le.LFIDtoNormal[3,2]
        le.L_tau4 = le.M_inv * le.chi_face[4]' * le.W_face[4] * le.LFIDtoNormal[4,2]
        le.D_tau = le.M_inv * le.chi_soln' * le.W_soln * le.d_phi_quad_d_eta
    end
    le.L_xi1 = le.MpK_inv * le.chi_face[1]' * le.W_face[1] * le.LFIDtoNormal[1,1]
    le.L_xi2 = le.MpK_inv * le.chi_face[2]' * le.W_face[2] * le.LFIDtoNormal[2,1]
    le.D_xi = le.MpK_inv * le.chi_soln' * le.W_soln * le.d_phi_quad_d_xi
end
