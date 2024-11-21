mutable struct DG
    # Category 1: inputs
    P::Int #polynomial degree
    dim::Int #dimension - note that dim must be 1 for now!
    N_elem_per_dim::Int
    N_state::Int # Number of states in the PDE
    domain_x_limits::Vector{Float64} # x-limits of rectangular domain

    #Category 2:defined from Category 1.
    N_elem::Int
    N_soln_per_dim::Int # Number of points per direction per cell
    N_soln::Int # Total number of points per cell = N_soln^dim 
    N_soln_dof::Int # Total number of DOFs per cell = N_soln*N_state
    N_soln_dof_global::Int # Global number of DOFs, i.e. length of the solution vector
    N_flux_per_dim::Int # Number of points per direction per cell
    N_flux::Int # Total number of points per cell = N_flux^dim 
    N_flux_dof::Int # Total number of DOFs per cell = N_flux*N_state #UNSURE IF THIS IS USED. 
    N_vol_per_dim::Int # WARNING: NOT FULLY IMPLEMENTED! Number of points per direction per cell. Used for unever # of basis and soln points.
    N_vol::Int # Total number of points per cell = N_flux^dim 
    N_faces::Int
    N_face::Int # points on each face. I assume that the face nodes are 1F flux nodes.
    VX::Vector{Float64} # Array of points defining the extremes of each element along one dimension
    delta_x::Float64 # length of evenly-spaced Cartesian elements
    x::Vector{Float64} # physical x coords, index are global ID.
    y::Vector{Float64} # physical y coords, index are global ID.
    
    GIDtoLID::Vector{Int} #Index is global ID, values are local IDs
    EIDLIDtoGID_basis::AbstractMatrix{Int} # Index of first dimension is element ID, index of second
                             # dimension is element ID. values are global ID.
    EIDLIDtoGID_vol::AbstractMatrix{Int} # Index of first dimension is element ID, index of second
                             # dimension is element ID. values are global ID.
    #LIDtoLFID::Vector{Int} # Index is local ID, value is local face ID
    #                       # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    LFIDtoLID::AbstractMatrix{Int} # Index is local face ID, values are LID corresponding to that face
    LFIDtoNormal::AbstractMatrix{Int} # Normal of LFID,
                                      # first column is x, second column is y
    #EIDLFIDtoGIDofexterior::AbstractMatrix{Int} # Linker to exterior value at a face.
    #                                    # Index of first dim is element ID, index of second 
    #                                    # dimension is LFID of the edge.
    EIDLFIDtoEIDofexterior::AbstractMatrix{Int} # Linker to exterior ELEM at a face.
                                                # Index of first dim is element ID, index of second 
                                                # dimension is LFID of the edge.
    LFIDtoLFIDofexterior::Vector{Int} # which LFID of the exterior cell matches to the index LFID.
    EIDtoTSID::Vector{Int} # maps element ID (index) to time-slab ID. 
                           # Elements with the same TSID are in the same "row"
                           # and cover the same t (y) vaues.
    TSIDtoEID::AbstractMatrix{Int} # TSID is the index, columns are EID.

    StIDLIDtoLSID::AbstractMatrix{Int} # StID is the state ID, 1:Nstate
                                       # LID is the ID of the node
                                       # LSID ("local storage") indicates the index in the storage vector
    StIDGIDtoGSID::AbstractMatrix{Int}
    #LXIDLYIDtoLID::AbstractMatrix{Int} # local x, y (length N_soln_per_dim) to local ID (length N_soln)
    #LIDtoLXIDLYID::AbstractMatrix{Int} # local x, y (length N_soln_per_dim) to local ID (length N_soln)


    r_soln::Vector{Float64}
    w_soln::Vector{Float64}
    r_basis::Vector{Float64}
    w_basis::Vector{Float64}
    r_flux::Vector{Float64}
    w_flux::Vector{Float64}
    #chi are basis functions for the solution.
    chi_soln::AbstractMatrix{Float64}
    chi_flux::AbstractMatrix{Float64}
    chi_face::AbstractArray{Float64}
    chi_vf::AbstractMatrix{Float64} # stacked matrices for skew-symmetric form
    #phi are basis functions for the flux.
    phi_flux::AbstractMatrix{Float64}
    phi_face::AbstractArray{Float64}
    d_phi_flux_d_xi::AbstractMatrix{Float64}
    d_phi_flux_d_eta::AbstractMatrix{Float64}
    C_m::AbstractMatrix{Float64}
    
    W_soln::AbstractMatrix{Float64}
    W_face::AbstractMatrix{Float64}
    W_flux::AbstractMatrix{Float64}
    J_soln::AbstractMatrix{Float64}
    J_face::AbstractMatrix{Float64}
    M::AbstractMatrix{Float64}
    MpK::AbstractMatrix{Float64}
    M_inv::AbstractMatrix{Float64}
    MpK_inv::AbstractMatrix{Float64}
    S_xi::AbstractMatrix{Float64}
    S_eta::AbstractMatrix{Float64}
    S_noncons_xi::AbstractMatrix{Float64}
    S_noncons_eta::AbstractMatrix{Float64}
    Pi_soln::AbstractMatrix{Float64}
    Pi_flux::AbstractMatrix{Float64}
    K::AbstractMatrix{Float64}
    QtildemQtildeT::AbstractArray{Float64}

    #Incomplete initializer - only assign Category 1 variables.
    DG(P::Int, 
       dim::Int, 
       N_elem_per_dim::Int,
       N_state::Int,
       domain_x_limits::Vector{Float64}) = new(P::Int,
                                               dim::Int,
                                               N_elem_per_dim::Int,
                                               N_state::Int,
                                               domain_x_limits::Vector{Float64})

end


function tensor_product_2D(A::AbstractMatrix, B::AbstractMatrix)
    # Modelled after tensor_product() function in PHiLiP
    # Returns C = A⊗ B

    rows_A = size(A)[1]
    rows_B = size(B)[1]
    cols_A = size(A)[2]
    cols_B = size(B)[2]
    
    C = zeros(Float64, (rows_A*rows_B, cols_A*cols_B))

    for j = 1:rows_B
        for k = 1:rows_A
            for n = 1:cols_B
                for o = 1:cols_A
                    irow = rows_A * (j-1) + k
                    icol = cols_A * (n-1) + o
                    C[irow, icol] = A[k,o] * B[j,n]
                end
            end
        end
    end

    return C
    
end


function build_coords_vectors(ref_vec_1D, dg::DG)

    x = zeros(dg.N_elem*(length(ref_vec_1D)^dg.dim))
    y = zeros(dg.N_elem*(length(ref_vec_1D)^dg.dim))
    Np=length(ref_vec_1D)^dg.dim
    Np_per_dim=length(ref_vec_1D)
    if dg.dim == 1
        for ielem = 1:dg.N_elem
            x_local = dg.VX[ielem] .+ 0.5* (ref_vec_1D .+1) * dg.delta_x
            x[(ielem - 1) * Np+1:ielem*Np] .= x_local
        end
    elseif dg.dim == 2
         for ielem = 1:dg.N_elem
            x_index = mod(ielem-1,dg.N_elem_per_dim)+1
            x_local_1D = dg.VX[x_index] .+ 0.5* (ref_vec_1D .+1) * dg.delta_x
            x_local = zeros(Np)
            for inode = 1:Np_per_dim
                #slightly gross indexing because we don't want to use LXIDLYIDtoLID for generality of ref_vec_1D.
                x_local[vec((1:Np_per_dim)' .+ (inode-1)*Np_per_dim)] .= x_local_1D
            end
            
            y_index = Int(ceil(ielem/dg.N_elem_per_dim))
            y_local_1D = dg.VX[y_index] .+ 0.5* (ref_vec_1D .+1) * dg.delta_x
            y_local = zeros(size(x_local))
            for inode = 1:Np_per_dim
                y_local[(1:Np_per_dim).*Np_per_dim.-(Np_per_dim-inode)] .= y_local_1D
            end

            x[(ielem - 1) * Np+1:ielem*Np] .= x_local
            y[(ielem - 1) * Np+1:ielem*Np] .= y_local
         end
    end
    return (x,y)
end

# Outer constructor for DG object. Might be good to move to inner constructor at some point.
function init_DG(P::Int, dim::Int, N_elem_per_dim::Int, N_state::Int, domain_x_limits::Vector{Float64},
        solnnodes::String, basisnodes::String, fluxnodes::String, fluxnodes_overintegration::Int, fluxreconstructionC::Float64,
        usespacetime::Bool)
    
    #initialize incomplete DG struct
    dg = DG(P, dim, N_elem_per_dim, N_state, domain_x_limits)
    dg.N_soln_per_dim = P+1
    dg.N_flux_per_dim = dg.N_soln_per_dim + fluxnodes_overintegration

    if dim == 1
        dg.N_soln = dg.N_soln_per_dim
        dg.N_flux = dg.N_flux_per_dim
        dg.N_faces = 2
        dg.N_face = 1
        dg.N_elem = N_elem_per_dim
    elseif dim == 2
        dg.N_faces = 4
        dg.N_soln = dg.N_soln_per_dim^dim
        dg.N_flux = dg.N_flux_per_dim^dim
        dg.N_face = dg.N_flux_per_dim
        dg.N_elem = N_elem_per_dim^dim
    end

    dg.N_soln_dof = dg.N_soln * dg.N_state
    dg.N_soln_dof_global = dg.N_soln_dof * dg.N_elem
    dg.N_vol_per_dim = dg.N_soln_per_dim # for testing, assume overintegrate by 1.
    dg.N_vol = dg.N_vol_per_dim^dim # for testing, assume overintegrate by 1.

    
    # Index is global ID, values are local IDs
    dg.GIDtoLID = mod.(0:(dg.N_soln*dg.N_elem.-1),dg.N_soln).+1
    
    # Index of first dimension is element ID, index of second dimension is element ID
    # values are global ID
    dg.EIDLIDtoGID_basis = reshape(1:dg.N_soln*dg.N_elem, (dg.N_soln,dg.N_elem))' #note transpose
    dg.EIDLIDtoGID_vol = reshape(1:dg.N_vol*dg.N_elem, (dg.N_vol,dg.N_elem))' #note transpose

    # Index is local ID, value is local face ID
    # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    if dim == 1
        dg.LFIDtoNormal = reshape([-1; 1], 2, 1) # normal of left face is 1, normal of right face is 1.
        dg.LFIDtoLID = reshape([1,dg.N_soln_per_dim], 2,1)
    elseif dim == 2
        dg.LFIDtoNormal = [-1 0; 1 0; 0 -1; 0 1] #first col. is x, second col. is y
        dg.LFIDtoLID = [(0:dg.N_soln_per_dim-1)' *dg.N_soln_per_dim.+1 ;
                        (1:dg.N_soln_per_dim)' *dg.N_soln_per_dim;
                        (1:dg.N_soln_per_dim)';
                        (1:dg.N_soln_per_dim)' .+ (dg.N_soln-dg.N_soln_per_dim)
                       ]
    end

    if dim == 1
        dg.EIDLFIDtoEIDofexterior = [circshift(1:dg.N_elem_per_dim,1)';circshift(1:dg.N_elem_per_dim,-1)']'
    elseif dim == 2
        dg.EIDLFIDtoEIDofexterior = zeros(Int, (dg.N_elem, dg.N_faces))
        for ielem = 1:dg.N_elem
            # face 1: left
            # regular joining (assume not on boundary)
            dg.EIDLFIDtoEIDofexterior[ielem,1] = ielem - 1
            if mod(ielem,dg.N_elem_per_dim) == 1
                #if on a periodic boundary
                dg.EIDLFIDtoEIDofexterior[ielem,1]+=dg.N_elem_per_dim
            end
            # face 2: right
            dg.EIDLFIDtoEIDofexterior[ielem,2] = ielem + 1
            if mod(ielem,dg.N_elem_per_dim) == 0
                dg.EIDLFIDtoEIDofexterior[ielem,2]-=dg.N_elem_per_dim
            end
            # face 3: bottom
            dg.EIDLFIDtoEIDofexterior[ielem,3] = ielem - dg.N_elem_per_dim
            if ielem/dg.N_elem_per_dim <= 1
                dg.EIDLFIDtoEIDofexterior[ielem,3] += dg.N_elem
                if usespacetime
                    # when using space-time, bottom is assigned dirichlet
                    # boundary
                    # This is set using an EIDofexterior = 0
                    dg.EIDLFIDtoEIDofexterior[ielem,3] = 0
                end
            end
            # face 4: top
            dg.EIDLFIDtoEIDofexterior[ielem,4] = ielem + dg.N_elem_per_dim
            if ielem/dg.N_elem_per_dim > dg.N_elem_per_dim - 1
                dg.EIDLFIDtoEIDofexterior[ielem,4] -= dg.N_elem
                if usespacetime
                    # when using space-time, top is outflow (transmissive)
                    # boundary
                    # This is set using an EIDofexterior = -1
                    dg.EIDLFIDtoEIDofexterior[ielem,4] = -1 
                end
            end
        end
    end
    dg.LFIDtoLFIDofexterior = [2, 1, 4, 3] #Hard-code for 2D. Also works fine for 1D. 
    #=== Not currently used anywhere (it messes with the generality of soln and basis nodes)
    if dim == 2
        dg.LXIDLYIDtoLID = zeros(Int, (N_soln_per_dim,N_soln_per_dim))
        dg.LIDtoLXIDLYID = zeros(Int, (N_soln_per_dim*N_soln_per_dim,2))
        counter=1
        for inodey = 1:N_soln_per_dim
            for inodex = 1:N_soln_per_dim
                dg.LXIDLYIDtoLID[inodex,inodey]=counter
                dg.LIDtoLXIDLYID[counter,:] .= [inodex,inodey]
                counter = counter+1
            end
        end
    end
    ===#
    if dim==2 #Only used for space-time, but the DG object has no knowledge of what is space-time
              # so we assemble for any 2D domain
        dg.EIDtoTSID = zeros(dg.N_elem)
        dg.TSIDtoEID = zeros(dg.N_elem_per_dim, dg.N_elem_per_dim)
        for ielem = 1:dg.N_elem
            iTS = convert(Int, floor((ielem-1)/dg.N_elem_per_dim+1))
            dg.EIDtoTSID[ielem] = iTS
            dg.TSIDtoEID[iTS, mod(ielem-1,dg.N_elem_per_dim)+1] = ielem
        end
    end

    dg.StIDLIDtoLSID = zeros(dg.N_state, dg.N_soln)
    ctr = 1
    for istate = 1:dg.N_state
        for ipoint = 1:dg.N_soln
            dg.StIDLIDtoLSID[istate,ipoint]=ctr
            ctr+=1
        end
    end

    dg.StIDGIDtoGSID = zeros(dg.N_state, dg.N_soln*dg.N_elem)
    ctr = 1
    for ielem = 1:dg.N_elem
        for istate = 1:dg.N_state
            for ipoint = 1:dg.N_soln
                dg.StIDGIDtoGSID[istate,ipoint+dg.N_soln*(ielem-1)]=ctr
                ctr+=1
            end
        end
    end


    # Solution nodes (integration nodes)
    if cmp(solnnodes, "GLL") == 0 
        display("GLL Volume nodes.")
        dg.r_soln,dg.w_soln = FastGaussQuadrature.gausslobatto(dg.N_vol_per_dim)
    elseif cmp(solnnodes, "GL") == 0
        display("GL Volume nodes.")
        dg.r_soln,dg.w_soln = FastGaussQuadrature.gaussjacobi(dg.N_vol_per_dim, 0.0,0.0)
    else
        display("Illegal soln node choice!")
    end
    # dg.r_soln= dg.r_soln * 0.5 .+ 0.5 # for changing ref element to match PHiLiP for debugging purposes
    # dg.w_soln /= 2.0

    # Basis function nodes (shape functions, interpolation nodes)
    if cmp(basisnodes, "GLL") == 0 
        display("GLL basis nodes.")
        dg.r_basis,dg.w_basis=FastGaussQuadrature.gausslobatto(dg.N_soln_per_dim)
    elseif cmp(basisnodes, "GL") == 0
        display("GL basis nodes.")
        dg.r_basis,dg.w_basis=FastGaussQuadrature.gaussjacobi(dg.N_soln_per_dim,0.0,0.0)
    else
        display("Illegal basis node choice!")
    end
    # dg.r_basis = dg.r_basis * 0.5 .+ 0.5
    # dg.w_basis /= 2.0
    
    # Flux nodes (shape functions, interpolation nodes)
    # Assume collocated flux basis nodes.
    if cmp(fluxnodes, "GLL") == 0 
        display("GLL flux nodes.")
        dg.r_flux,dg.w_flux=FastGaussQuadrature.gausslobatto(dg.N_soln_per_dim+fluxnodes_overintegration)
    elseif cmp(fluxnodes, "GL") == 0
        display("GL flux nodes.")
        dg.r_flux,dg.w_flux=FastGaussQuadrature.gaussjacobi(dg.N_soln_per_dim+fluxnodes_overintegration,0.0,0.0)
    else
        display("Illegal flux node choice!")
    end
    if fluxnodes_overintegration>0
        display("Overintegrating the flux by")
        display(fluxnodes_overintegration)
    end
    # dg.r_basis = dg.r_basis * 0.5 .+ 0.5
    # dg.w_basis /= 2.0

    dg.VX = range(domain_x_limits[1],domain_x_limits[2], dg.N_elem_per_dim+1) |> collect
    display("Elements per dim:")
    display(dg.N_elem_per_dim)
    dg.delta_x = dg.VX[2]-dg.VX[1]
    # constant jacobian on all elements as they are evenly spaced
    jacobian = (dg.delta_x/2.0)^dim #reference element is 2 units long
    # jacobian = (dg.delta_x/1.0)^dim
    dg.J_soln = LinearAlgebra.diagm(ones(length(dg.r_soln)^dim)*jacobian)

    (dg.x, dg.y) = build_coords_vectors(dg.r_soln, dg) 
    # Define Vandermonde matrices
    if dim == 1
        dg.chi_soln = vandermonde1D(dg.r_soln,dg.r_basis)
        dg.phi_flux = vandermonde1D(dg.r_flux,dg.r_flux)
        dg.d_phi_flux_d_xi = gradvandermonde1D(dg.r_flux,dg.r_flux)
        d_chi_flux_d_xi = gradvandermonde1D(dg.r_flux,dg.r_basis)
        #reference coordinates of L and R faces
        r_f_L::Float64 = -1
        r_f_R::Float64 = 1
        dg.chi_face = assembleFaceVandermonde1D(r_f_L,r_f_R,dg.r_basis)
        dg.phi_face = assembleFaceVandermonde1D(r_f_L,r_f_R,dg.r_flux)
        dg.chi_flux = vandermonde1D(dg.r_flux,dg.r_basis)

        dg.W_soln = LinearAlgebra.diagm(dg.w_soln) # diagonal matrix holding quadrature weights
        dg.W_flux = LinearAlgebra.diagm(dg.w_flux) # diagonal matrix holding quadrature weights
        dg.W_face = reshape([1.0], 1, 1) #1x1 matrix for generality with higher dim
        dg.C_m = reshape([1.0], 1, 1) #1x1 matrix for generality with higher dim
    elseif dim == 2
        dg.chi_soln = vandermonde2D(dg.r_soln,dg.r_basis, dg)
        dg.chi_flux = vandermonde2D(dg.r_flux,dg.r_basis, dg)
        dg.phi_flux = vandermonde2D(dg.r_flux,dg.r_flux,dg)
        dg.d_phi_flux_d_xi = gradvandermonde2D(1, dg.r_flux,dg.r_flux, dg)
        dg.d_phi_flux_d_eta = gradvandermonde2D(2, dg.r_flux,dg.r_flux, dg)
        d_chi_flux_d_xi = gradvandermonde2D(1, dg.r_flux,dg.r_basis, dg)
        d_chi_flux_d_eta = gradvandermonde2D(2, dg.r_flux,dg.r_basis, dg)
        dg.chi_face = assembleFaceVandermonde2D(dg.r_basis,dg.r_flux,dg) #face nodes are 1D flux nodes
        dg.phi_face = assembleFaceVandermonde2D(dg.r_flux,dg.r_flux,dg)
        dg.W_soln = LinearAlgebra.diagm(vec(dg.w_soln*dg.w_soln'))
        dg.W_flux = LinearAlgebra.diagm(vec(dg.w_flux*dg.w_flux'))
        dg.W_face = LinearAlgebra.diagm(dg.w_flux)
        dg.J_face = LinearAlgebra.diagm(ones(length(dg.r_flux)) * jacobian ^ (1/dim)) # 1D jacobian on the face of the element.
        dg.C_m = dg.delta_x/2.0 * [1 0; 0 1]  # Assuming a cartesian element and a reference element (-1,1)
    end

    # for skew-symmetric stiffness operator form
    dg.chi_vf = dg.chi_flux'
    for iface=1:dg.N_faces
        dg.chi_vf = [dg.chi_vf dg.chi_face[:,:,iface]' ]
    end
    
    # Mass and stiffness matrices as defined Eq. 9.5 Cicchino 2022
    # All defined on a single element.
    dg.M = dg.chi_soln' * dg.W_soln * dg.J_soln * dg.chi_soln ## Have verified this against PHiLiP. Here, unmodified mass matrix.
    dg.S_xi = dg.chi_flux' * dg.W_flux * dg.d_phi_flux_d_xi
    dg.S_noncons_xi = dg.W_flux * d_chi_flux_d_xi
    if dim==2
        dg.S_eta = dg.chi_flux' * dg.W_flux * dg.d_phi_flux_d_eta
        dg.S_noncons_eta = dg.W_flux * d_chi_flux_d_eta
    end
    M_nojac_soln = dg.chi_soln' * dg.W_soln * dg.chi_soln
    dg.Pi_soln = inv(M_nojac_soln)*dg.chi_soln'*dg.W_soln
    dg.K = zeros(size(dg.M))
    if dim==1 && fluxreconstructionC != 0.0
        d_chi_soln_d_xi = gradvandermonde1D(dg.r_soln,dg.r_basis)
        D_xi = inv(M_nojac_soln) * dg.chi_soln' * dg.W_soln * d_chi_soln_d_xi
        dg.K = fluxreconstructionC * ( (D_xi)^P )' * dg.M * ( (D_xi)^P ) ## Verified for 1D.
    elseif dim==2 && fluxreconstructionC != 0.0
        #==== This version uses tensor products
        dg.K = 0*dg.M #initialize with size of M

        chi_v_1D = vandermonde1D(dg.r_soln,dg.r_basis)
        d_chi_v_d_xi_1D = gradvandermonde1D(dg.r_soln,dg.r_basis)
        W_1D = dg.W_f
        J_1D = dg.J_f
        M_1D = chi_v_1D' * W_1D * J_1D * chi_v_1D
        display(M_1D) # ok
        D_xi_1D_P = (inv(chi_v_1D' * W_1D * chi_v_1D) * (chi_v_1D' * W_1D * d_chi_v_d_xi_1D))^P
        display(D_xi_1D_P) # NOT okay.

        K_1D = fluxreconstructionC * (D_xi_1D_P)' * M_1D * D_xi_1D_P
        
        FR1 = tensor_product_2D(K_1D, M_1D)
        FR2 = tensor_product_2D(M_1D, K_1D)
        FRcross = tensor_product_2D(K_1D, K_1D)

        dg.K = FR1 + FR2 + FRcross


        Following code calculates per Cicchino 2022 curvilinear eq. 28-29. Does the same thing as above code.
        ==# 
        d_chi_soln_d_xi = gradvandermonde2D(1, dg.r_soln, dg.r_basis, dg)
        d_chi_soln_d_eta = gradvandermonde2D(2, dg.r_soln, dg.r_basis, dg)
        D_xi = inv(M_nojac_soln)*dg.chi_soln' * dg.W_soln * d_chi_soln_d_xi
        D_eta = inv(M_nojac_soln)*dg.chi_soln' * dg.W_soln * d_chi_soln_d_eta
        dg.K = 0*dg.M #initialize with size of M
        if usespacetime
            # K only in space
            dg.K = fluxreconstructionC * (D_xi^P)' * dg.M * D_xi^P
        else
            # K in space and in time
            for s = [0,P]
                for v = [0,P]
                    if s+v >= P
                        c_sv = fluxreconstructionC ^ ((s/P) + (v/P))
                        dg.K += c_sv * (D_xi^s * D_eta^v)' * dg.M * (D_xi^s * D_eta^v)
                    end
                end
            end
        end
    end
    display("FR K")
    display(dg.K) # Verified against PHiLiP for 1D and 2D using C from PHiLiP


    dg.M_inv = inv(dg.M) # unmodified mass matrix.
    dg.MpK = dg.M + dg.K # Here, modified mass matrix.
    display("Adjusted Mass matrix")
    display(dg.MpK)
    dg.MpK_inv = inv(dg.MpK)

    M_nojac_flux = dg.phi_flux' * dg.W_flux * dg.phi_flux
    dg.Pi_flux = inv(M_nojac_flux)*dg.phi_flux'*dg.W_flux

    Q_dimension = dg.N_flux+dg.N_faces*dg.N_face # N_flux points in soln, Nfp on each face. Store ndim matrices.
    dg.QtildemQtildeT = zeros(Q_dimension,Q_dimension,dim) 
    # soln
    dg.QtildemQtildeT[1:dg.N_flux, 1:dg.N_flux,1] .= dg.W_flux * dg.d_phi_flux_d_xi- dg.d_phi_flux_d_xi' * dg.W_flux
    if dim==2
        dg.QtildemQtildeT[1:dg.N_flux, 1:dg.N_flux,2] .= dg.W_flux * dg.d_phi_flux_d_eta- dg.d_phi_flux_d_eta' * dg.W_flux
    end

    # face - only assemble top-right matrix
    for iface = 1:dg.N_faces
        dg.QtildemQtildeT[1:dg.N_flux,(dg.N_flux+1+dg.N_face*(iface-1)):(dg.N_flux+dg.N_face*iface),1] .+= dg.phi_face[:,:,iface]' * dg.W_face * dg.LFIDtoNormal[iface,1] # 1st direction of normal
    end
    if dim==2
        for iface = 1:dg.N_faces
            dg.QtildemQtildeT[1:dg.N_flux,(dg.N_flux+1+dg.N_face*(iface-1)):(dg.N_flux+dg.N_face*iface),2] .+= dg.phi_face[:,:,iface]' * dg.W_face * dg.LFIDtoNormal[iface,2] # 2nd direction of normal
        end
    end
    # then assign skew-symmetric matrix
    for idim = 1:dim
        dg.QtildemQtildeT[dg.N_flux+1:end, 1:dg.N_flux, idim] .=  -1.0 * dg.QtildemQtildeT[1:dg.N_flux,dg.N_flux+1:end,idim]'
    end

    #display("Skew-symmetric stiffness operator:")
    #display(dg.QtildemQtildeT)

    return dg

end

