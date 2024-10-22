mutable struct DG
    # Category 1: inputs
    P::Int #polynomial degree
    dim::Int #dimension - note that dim must be 1 for now!
    N_elem_per_dim::Int
    N_state::Int # Number of states in the PDE
    domain_x_limits::Vector{Float64} # x-limits of rectangular domain

    #Category 2:defined from Category 1.
    N_elem::Int
    Np_per_dim::Int # Number of points per direction per cell
    Np::Int # Total number of points per cell = Np^dim 
    N_dof::Int # Total number of DOFs per cell = Np*N_state
    N_dof_global::Int # Global number of DOFs, i.e. length of the solution vector
    N_vol_per_dim::Int # Number of points per direction per cell
    N_vol::Int # Total number of points per cell = Np^dim 
    Nfaces::Int
    Nfp::Int
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
    #LXIDLYIDtoLID::AbstractMatrix{Int} # local x, y (length Np_per_dim) to local ID (length Np)
    #LIDtoLXIDLYID::AbstractMatrix{Int} # local x, y (length Np_per_dim) to local ID (length Np)


    r_volume::Vector{Float64}
    w_volume::Vector{Float64}
    r_basis::Vector{Float64}
    w_basis::Vector{Float64}
    chi_v::AbstractMatrix{Float64}
    d_chi_v_d_xi::AbstractMatrix{Float64}
    d_chi_v_d_eta::AbstractMatrix{Float64}
    chi_f::AbstractArray{Float64}
    chi_vf::AbstractMatrix{Float64}
    C_m::AbstractMatrix{Float64}
    
    W::AbstractMatrix{Float64}
    W_f::AbstractMatrix{Float64}
    J::AbstractMatrix{Float64}
    J_f::AbstractMatrix{Float64}
    M::AbstractMatrix{Float64}
    MpK::AbstractMatrix{Float64}
    M_inv::AbstractMatrix{Float64}
    MpK_inv::AbstractMatrix{Float64}
    S_xi::AbstractMatrix{Float64}
    S_eta::AbstractMatrix{Float64}
    S_noncons_xi::AbstractMatrix{Float64}
    S_noncons_eta::AbstractMatrix{Float64}
    M_nojac::AbstractMatrix{Float64}
    Pi::AbstractMatrix{Float64}
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
        volumenodes::String, basisnodes::String, fluxreconstructionC::Float64,
        usespacetime::Bool)
    
    #initialize incomplete DG struct
    dg = DG(P, dim, N_elem_per_dim, N_state, domain_x_limits)
    dg.Np_per_dim = P+1

    if dim == 1
        dg.Np=dg.Np_per_dim
        dg.Nfaces = 2
        dg.Nfp = 1
        dg.N_elem = N_elem_per_dim
    elseif dim == 2
        dg.Nfaces = 4
        dg.Np=dg.Np_per_dim^dim
        dg.Nfp = dg.Np_per_dim
        dg.N_elem = N_elem_per_dim^dim
    end
    Np_per_dim=P+1 # for convenience
    N_elem = dg.N_elem
    Np = dg.Np
    dg.N_dof = dg.Np * dg.N_state
    dg.N_dof_global = dg.N_dof * dg.N_elem
    dg.N_vol_per_dim = dg.Np_per_dim # for testing, assume overintegrate by 1.
    dg.N_vol = dg.N_vol_per_dim^dim # for testing, assume overintegrate by 1.

    
    # Index is global ID, values are local IDs
    dg.GIDtoLID = mod.(0:(dg.Np*dg.N_elem.-1),dg.Np).+1
    
    # Index of first dimension is element ID, index of second dimension is element ID
    # values are global ID
    dg.EIDLIDtoGID_basis = reshape(1:Np*N_elem, (Np,N_elem))' #note transpose
    dg.EIDLIDtoGID_vol = reshape(1:dg.N_vol*N_elem, (dg.N_vol,N_elem))' #note transpose

    # Index is local ID, value is local face ID
    # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    if dim == 1
        dg.LFIDtoNormal = reshape([-1; 1], 2, 1) # normal of left face is 1, normal of right face is 1.
        dg.LFIDtoLID = reshape([1,Np_per_dim], 2,1)
    elseif dim == 2
        dg.LFIDtoNormal = [-1 0; 1 0; 0 -1; 0 1] #first col. is x, second col. is y
        dg.LFIDtoLID = [(0:Np_per_dim-1)' *Np_per_dim.+1 ;
                        (1:Np_per_dim)' *Np_per_dim;
                        (1:Np_per_dim)';
                        (1:Np_per_dim)' .+ (Np-Np_per_dim)
                       ]
    end

    if dim == 1
        dg.EIDLFIDtoEIDofexterior = [circshift(1:N_elem_per_dim,1)';circshift(1:N_elem_per_dim,-1)']'
    elseif dim == 2
        dg.EIDLFIDtoEIDofexterior = zeros(Int, (N_elem, dg.Nfaces))
        for ielem = 1:N_elem
            # face 1: left
            # regular joining (assume not on boundary)
            dg.EIDLFIDtoEIDofexterior[ielem,1] = ielem - 1
            if mod(ielem,N_elem_per_dim) == 1
                #if on a periodic boundary
                dg.EIDLFIDtoEIDofexterior[ielem,1]+=N_elem_per_dim
            end
            # face 2: right
            dg.EIDLFIDtoEIDofexterior[ielem,2] = ielem + 1
            if mod(ielem,N_elem_per_dim) == 0
                dg.EIDLFIDtoEIDofexterior[ielem,2]-=N_elem_per_dim
            end
            # face 3: bottom
            dg.EIDLFIDtoEIDofexterior[ielem,3] = ielem - N_elem_per_dim
            if ielem/N_elem_per_dim <= 1
                dg.EIDLFIDtoEIDofexterior[ielem,3] += N_elem
                if usespacetime
                    # when using space-time, bottom is assigned dirichlet
                    # boundary
                    # This is set using an EIDofexterior = 0
                    dg.EIDLFIDtoEIDofexterior[ielem,3] = 0
                end
            end
            # face 4: top
            dg.EIDLFIDtoEIDofexterior[ielem,4] = ielem + N_elem_per_dim
            if ielem/N_elem_per_dim > N_elem_per_dim - 1
                dg.EIDLFIDtoEIDofexterior[ielem,4] -= N_elem
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
    #=== Not currently used anywhere (it messes with the generality of volume and basis nodes)
    if dim == 2
        dg.LXIDLYIDtoLID = zeros(Int, (Np_per_dim,Np_per_dim))
        dg.LIDtoLXIDLYID = zeros(Int, (Np_per_dim*Np_per_dim,2))
        counter=1
        for inodey = 1:Np_per_dim
            for inodex = 1:Np_per_dim
                dg.LXIDLYIDtoLID[inodex,inodey]=counter
                dg.LIDtoLXIDLYID[counter,:] .= [inodex,inodey]
                counter = counter+1
            end
        end
    end
    ===#
    if dim==2 #Only used for space-time, but the DG object has no knowledge of what is space-time
              # so we assemble for any 2D domain
        dg.EIDtoTSID = zeros(N_elem)
        dg.TSIDtoEID = zeros(N_elem_per_dim, N_elem_per_dim)
        for ielem = 1:N_elem
            iTS = convert(Int, floor((ielem-1)/N_elem_per_dim+1))
            dg.EIDtoTSID[ielem] = iTS
            dg.TSIDtoEID[iTS, mod(ielem-1,N_elem_per_dim)+1] = ielem
        end
    end

    dg.StIDLIDtoLSID = zeros(dg.N_state, dg.Np)
    ctr = 1
    for istate = 1:dg.N_state
        for ipoint = 1:dg.Np
            dg.StIDLIDtoLSID[istate,ipoint]=ctr
            ctr+=1
        end
    end
    display(dg.StIDLIDtoLSID)

    dg.StIDGIDtoGSID = zeros(dg.N_state, dg.Np*dg.N_elem)
    ctr = 1
    for ielem = 1:dg.N_elem
        for istate = 1:dg.N_state
            for ipoint = 1:dg.Np
                dg.StIDGIDtoGSID[istate,ipoint+Np*(ielem-1)]=ctr
                ctr+=1
            end
        end
    end
    display(dg.StIDGIDtoGSID)


    # Solution nodes (integration nodes)
    if cmp(volumenodes, "GLL") == 0 
        display("GLL Volume nodes.")
        dg.r_volume,dg.w_volume = FastGaussQuadrature.gausslobatto(dg.N_vol_per_dim)
    elseif cmp(volumenodes, "GL") == 0
        display("GL Volume nodes.")
        dg.r_volume,dg.w_volume = FastGaussQuadrature.gaussjacobi(dg.N_vol_per_dim, 0.0,0.0)
    else
        display("Illegal volume node choice!")
    end
    # dg.r_volume= dg.r_volume * 0.5 .+ 0.5 # for changing ref element to match PHiLiP for debugging purposes
    # dg.w_volume /= 2.0

    # Basis function nodes (shape functions, interpolation nodes)
    if cmp(basisnodes, "GLL") == 0 
        display("GLL basis nodes.")
        dg.r_basis,dg.w_basis=FastGaussQuadrature.gausslobatto(dg.Np_per_dim)
    elseif cmp(basisnodes, "GL") == 0
        display("GL basis nodes.")
        dg.r_basis,dg.w_basis=FastGaussQuadrature.gaussjacobi(Np_per_dim,0.0,0.0)
    else
        display("Illegal basis node choice!")
    end
    # dg.r_basis = dg.r_basis * 0.5 .+ 0.5
    # dg.w_basis /= 2.0

    dg.VX = range(domain_x_limits[1],domain_x_limits[2], N_elem_per_dim+1) |> collect
    display("Elements per dim:")
    display(N_elem_per_dim)
    dg.delta_x = dg.VX[2]-dg.VX[1]
    # constant jacobian on all elements as they are evenly spaced
    jacobian = (dg.delta_x/2.0)^dim #reference element is 2 units long
    # jacobian = (dg.delta_x/1.0)^dim
    dg.J = LinearAlgebra.diagm(ones(length(dg.r_volume)^dim)*jacobian)

    (dg.x, dg.y) = build_coords_vectors(dg.r_volume, dg) 
    # Define Vandermonde matrices
    if dim == 1
        dg.chi_v = vandermonde1D(dg.r_volume,dg.r_basis)
        dg.d_chi_v_d_xi = gradvandermonde1D(dg.r_volume,dg.r_basis)
        #reference coordinates of L and R faces
        r_f_L::Float64 = -1
        r_f_R::Float64 = 1
        dg.chi_f = assembleFaceVandermonde1D(r_f_L,r_f_R,dg.r_basis)

        dg.W = LinearAlgebra.diagm(dg.w_volume) # diagonal matrix holding quadrature weights
        dg.W_f = reshape([1.0], 1, 1) #1x1 matrix for generality with higher dim
        dg.C_m = reshape([1.0], 1, 1) #1x1 matrix for generality with higher dim
    elseif dim == 2
        dg.chi_v = vandermonde2D(dg.r_volume,dg.r_basis, dg)
        dg.d_chi_v_d_xi = gradvandermonde2D(1, dg.r_volume,dg.r_basis, dg)
        dg.d_chi_v_d_eta = gradvandermonde2D(2, dg.r_volume,dg.r_basis, dg)
        dg.W = LinearAlgebra.diagm(vec(dg.w_volume*dg.w_volume'))
        dg.chi_f = assembleFaceVandermonde2D(dg.r_basis,dg.r_volume,dg)
        dg.W_f = LinearAlgebra.diagm(dg.w_volume)
        dg.J_f = LinearAlgebra.diagm(ones(length(dg.r_volume)) * jacobian ^ (1/dim)) # 1D jacobian on the face of the element.
        dg.C_m = dg.delta_x/2.0 * [1 0; 0 1]  # Assuming a cartesian element and a reference element (-1,1)
    end

    # for skew-symmetric stiffness operator form
    dg.chi_vf = dg.chi_v'
    for iface=1:dg.Nfaces
        dg.chi_vf = [dg.chi_vf dg.chi_f[:,:,iface]' ]
    end
    
    # Mass and stiffness matrices as defined Eq. 9.5 Cicchino 2022
    # All defined on a single element.
    dg.M = dg.chi_v' * dg.W * dg.J * dg.chi_v ## Have verified this against PHiLiP. Here, unmodified mass matrix.
    dg.S_xi = dg.chi_v' * dg.W * dg.d_chi_v_d_xi
    dg.S_noncons_xi = dg.W * dg.d_chi_v_d_xi
    if dim==2
        dg.S_eta = dg.chi_v' * dg.W * dg.d_chi_v_d_eta
        dg.S_noncons_eta = dg.W * dg.d_chi_v_d_eta
    end
    dg.M_nojac = dg.chi_v' * dg.W * dg.chi_v
    dg.Pi = inv(dg.M_nojac)*dg.chi_v'*dg.W
    if dim==1
        dg.K = fluxreconstructionC * ( (inv(dg.M_nojac)*dg.S_xi)^P )' * dg.M * ( (inv(dg.M_nojac)*dg.S_xi)^P ) ## Verified for 1D.
    elseif dim==2
        #==== This version uses tensor products
        dg.K = 0*dg.M #initialize with size of M

        chi_v_1D = vandermonde1D(dg.r_volume,dg.r_basis)
        d_chi_v_d_xi_1D = gradvandermonde1D(dg.r_volume,dg.r_basis)
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
        D_xi = inv(dg.M_nojac)*dg.S_xi
        D_eta = inv(dg.M_nojac)*dg.S_eta
        dg.K = 0*dg.M #initialize with size of M
        for s = [0,P]
            for v = [0,P]
                if s+v >= P
                    c_sv = fluxreconstructionC ^ ((s/P) + (v/P))
                    dg.K += c_sv * (D_xi^s * D_eta^v)' * dg.M * (D_xi^s * D_eta^v)
                end
            end
        end
        dg.K = fluxreconstructionC * (D_xi^P)' * dg.M * D_xi^P
    end
    display("FR K")
    display(dg.K) # Verified against PHiLiP for 1D and 2D using C from PHiLiP


    dg.M_inv = inv(dg.M) # unmodified mass matrix.
    dg.MpK = dg.M + dg.K # Here, modified mass matrix.
    display("Adjusted Mass matrix")
    display(dg.MpK)
    dg.MpK_inv = inv(dg.MpK)

    dg.QtildemQtildeT = zeros(Np+dg.Nfaces*dg.Nfp, Np+dg.Nfaces*dg.Nfp,dim) # Np points in volume, Nfp on each face. Store ndim matrices.
    # volume
    dg.QtildemQtildeT[1:Np, 1:Np,1] .= dg.W * dg.d_chi_v_d_xi- dg.d_chi_v_d_xi' * dg.W
    if dim==2
        dg.QtildemQtildeT[1:Np, 1:Np,2] .= dg.W * dg.d_chi_v_d_eta- dg.d_chi_v_d_eta' * dg.W
    end

    # face - only assemble top-right matrix
    for iface = 1:dg.Nfaces
        dg.QtildemQtildeT[1:Np,(Np+1+dg.Nfp*(iface-1)):(Np+dg.Nfp*iface),1] .+= dg.chi_f[:,:,iface]' * dg.W_f * dg.LFIDtoNormal[iface,1] # 1st direction of normal
    end
    if dim==2
        for iface = 1:dg.Nfaces
            dg.QtildemQtildeT[1:Np,(Np+1+dg.Nfp*(iface-1)):(Np+dg.Nfp*iface),2] .+= dg.chi_f[:,:,iface]' * dg.W_f * dg.LFIDtoNormal[iface,2] # 2nd direction of normal
        end
    end
    # then assign skew-symmetric matrix
    for idim = 1:dim
        dg.QtildemQtildeT[Np+1:end, 1:Np, idim] .=  -1.0 * dg.QtildemQtildeT[1:Np,Np+1:end,idim]'
    end

    display("Skew-symmetric stiffness operator:")
    display(dg.QtildemQtildeT)



    display("Next line is dg.chi_v*dg.Pi, which should be identity.")
    display(dg.chi_v*dg.Pi) #should be identity


    return dg

end

