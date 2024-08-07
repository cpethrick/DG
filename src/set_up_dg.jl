mutable struct DG
    # Category 1: inputs
    P::Int #polynomial degree
    dim::Int #dimension - note that dim must be 1 for now!
    N_elem_per_dim::Int
    domain_x_limits::Vector{Float64} # x-limits of rectangular domain

    #Category 2:defined from Category 1.
    N_elem::Int
    Np_per_dim::Int # Number of points per direction per cell
    Np::Int # Total number of points per cell = Np^dim 
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
    C_m::AbstractMatrix{Float64}
    
    W::AbstractMatrix{Float64}
    W_f::AbstractMatrix{Float64}
    J::AbstractMatrix{Float64}
    M::AbstractMatrix{Float64}
    M_inv::AbstractMatrix{Float64}
    S_xi::AbstractMatrix{Float64}
    S_eta::AbstractMatrix{Float64}
    S_noncons_xi::AbstractMatrix{Float64}
    S_noncons_eta::AbstractMatrix{Float64}
    M_nojac::AbstractMatrix{Float64}
    Pi::AbstractMatrix{Float64}

    #Incomplete initializer - only assign Category 1 variables.
    DG(P::Int, 
       dim::Int, 
       N_elem_per_dim::Int,
       domain_x_limits::Vector{Float64}) = new(P::Int,
                                               dim::Int,
                                               N_elem_per_dim::Int,
                                               domain_x_limits::Vector{Float64})

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
function init_DG(P::Int, dim::Int, N_elem_per_dim::Int,domain_x_limits::Vector{Float64}, volumenodes::String, basisnodes::String)
    
    #initialize incomplete DG struct
    dg = DG(P, dim, N_elem_per_dim, domain_x_limits)
    dg.Np_per_dim = P+1
    

    if dim == 1
        dg.Np=dg.Np_per_dim
        dg.Nfaces = 2
        dg.Nfp = 2
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
    #dg.LIDtoLFID = zeros(Int64,Np_per_dim)
    #dg.LIDtoLFID[[1,Np_per_dim]] .= 1:dg.Nfaces
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
            # regular joining (assume not on boundary)
            dg.EIDLFIDtoEIDofexterior[ielem,1] = ielem - 1
            if mod(ielem,N_elem_per_dim) == 1
                #if on a periodic boundary
                dg.EIDLFIDtoEIDofexterior[ielem,1]+=N_elem_per_dim
            end
            dg.EIDLFIDtoEIDofexterior[ielem,2] = ielem + 1
            if mod(ielem,N_elem_per_dim) == 0
                dg.EIDLFIDtoEIDofexterior[ielem,2]-=N_elem_per_dim
            end
            dg.EIDLFIDtoEIDofexterior[ielem,3] = ielem - N_elem_per_dim
            if ielem/N_elem_per_dim <= 1
                dg.EIDLFIDtoEIDofexterior[ielem,3] += N_elem
            end
            dg.EIDLFIDtoEIDofexterior[ielem,4] = ielem + N_elem_per_dim
            if ielem/N_elem_per_dim > N_elem_per_dim - 1
                dg.EIDLFIDtoEIDofexterior[ielem,4] -= N_elem
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
    display("r_volume")
    # dg.r_volume= dg.r_volume * 0.5 .+ 0.5 # for changing ref element to match PHiLiP for debugging purposes
    # dg.w_volume /= 2.0
    display(dg.r_volume)

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
    display("r_basis")
    # dg.r_basis = dg.r_basis * 0.5 .+ 0.5
    # dg.w_basis /= 2.0
    display(dg.r_basis)

    dg.VX = range(domain_x_limits[1],domain_x_limits[2], N_elem_per_dim+1) |> collect
    display(N_elem_per_dim)
    dg.delta_x = dg.VX[2]-dg.VX[1]
    display(dg.VX)
    # constant jacobian on all elements as they are evenly spaced
    jacobian = (dg.delta_x/2.0)^dim #reference element is 2 units long
    display("jacobian")
    display(jacobian)
    dg.J = LinearAlgebra.diagm(ones(length(dg.r_volume)^dim)*jacobian)

    (dg.x, dg.y) = build_coords_vectors(dg.r_volume, dg) 
    # Define Vandermonde matrices
    if dim == 1
        dg.chi_v = vandermonde1D(dg.r_volume,dg.r_basis)
        display(dg.chi_v)
        dg.d_chi_v_d_xi = gradvandermonde1D(dg.r_volume,dg.r_basis)
        display(dg.d_chi_v_d_xi)
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
        display("W")
        display(dg.W)
        display("1D w")
        display(dg.w_volume)
        dg.chi_f = assembleFaceVandermonde2D(dg.r_basis,dg.r_volume,dg)
        dg.W_f = LinearAlgebra.diagm(dg.w_volume)
        dg.C_m = dg.delta_x/2.0 * [1 0; 0 1]  # Assuming a cartesian element and a reference element (-1,1)
    end
    
    # Mass and stiffness matrices as defined Eq. 9.5 Cicchino 2022
    # All defined on a single element.
    dg.M = dg.chi_v' * dg.W * dg.J * dg.chi_v ## Have verified this against PHiLiP.
    display("Mass matrix")
    display(dg.M)
    dg.M_inv = inv(dg.M)
    dg.S_xi = dg.chi_v' * dg.W * dg.d_chi_v_d_xi
    dg.S_noncons_xi = dg.W * dg.d_chi_v_d_xi
    if dim==2
        dg.S_eta = dg.chi_v' * dg.W * dg.d_chi_v_d_eta
        dg.S_noncons_eta = dg.W * dg.d_chi_v_d_eta
    end
    dg.M_nojac = dg.chi_v' * dg.W * dg.chi_v
    dg.Pi = inv(dg.M_nojac)*dg.chi_v'*dg.W

    display("Next line is dg.chi_v*dg.Pi, which should be identity.")
    display(dg.chi_v*dg.Pi) #should be identity

    return dg

end

