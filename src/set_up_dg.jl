mutable struct DG
    # Category 1: inputs
    P::Int #polynomial degree
    dim::Int #dimension - note that dim must be 1 for now!
    N_elem_per_dim::Int
    domain_x_limits::Vector{Float64} # x-limits of rectangular domain

    #Category 2:defined from Category 1.
    Np::Int
    Nfaces::Int
    Nfp::Int
    VX::Vector{Float64} # Array of points defining the extremes of each element 
    delta_x::Float64 # length of evenly-spaced Cartesian elements
    x::Vector{Float64} # physical x coords, index are global ID.
    
    GIDtoLID::Vector{Int} #Index is global ID, values are local IDs
    EIDLIDtoGID::AbstractMatrix{Int} # Index of first dimension is element ID, index of second
                             # dimension is element ID. values are global ID.
    LIDtoLFID::Vector{Int} # Index is local ID, value is local face ID
                           # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    LFIDtoNormal::Vector{Int} # normal of left face is 1, normal of right face is 1.
    EIDLFIDtoGIDofexterior::AbstractMatrix{Int} # Linker to exterior value at a face.
                                        # Index of first dim is element ID, index of second 
                                        # dimension is LFID of the edge.

    r_volume::Vector{Float64}
    w::Vector{Float64}
    r_basis::Vector{Float64}
    w_basis::Vector{Float64}
    chi_v::AbstractMatrix{Float64}
    d_chi_v_d_xi::AbstractMatrix{Float64}
    chi_f::AbstractArray{Float64}
    
    W::AbstractMatrix{Float64}
    W_f::Float64 # AbstractMatrix{Float64}
    J::AbstractMatrix{Float64}
    M::AbstractMatrix{Float64}
    M_inv::AbstractMatrix{Float64}
    S_xi::AbstractMatrix{Float64}
    S_noncons::AbstractMatrix{Float64}
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

# Outer constructor for DG object. Might be good to move to inner constructor at some point.
function init_DG(P, dim, N_elem_per_dim,domain_x_limits)
    
    #initialize incomplete DG struct
    dg = DG(P, dim, N_elem_per_dim, domain_x_limits)
    Np=P+1 # for convenience
    dg.Np = P+1
    
    display(dg.P)

    if dim == 1
        dg.Nfaces = 2
        dg.Nfp = 2
    else
        display("Dim > 1 not implemented!")
    end
    
    # Index is global ID, values are local IDs
    dg.GIDtoLID = mod.(0:(Np*N_elem_per_dim.-1),Np).+1
    
    # Index of first dimension is element ID, index of second dimension is element ID
    # values are global ID
    dg.EIDLIDtoGID = reshape(1:Np*N_elem_per_dim, (Np,N_elem_per_dim))' #note transpose

    # Index is local ID, value is local face ID
    # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    dg.LIDtoLFID = zeros(Int64,Np)
    dg.LIDtoLFID[[1,Np]] .= 1:dg.Nfaces

    dg.LFIDtoNormal = [-1,1] # normal of left face is 1, normal of right face is 1.

    # Solution nodes - GLL
    # must choose GLL nodes unless I modify the selection of uP and uP for numerical flux.
    # Also will need to change splitting on the face.
    dg.r_volume,dg.w = FastGaussQuadrature.gausslobatto(dg.Np)
    #r will be size N+1

    # Basis function nodes - GLL
    # Note: must choose GLL due to face flux splitting.
    #r_basis,w_basis=FastGaussQuadrature.gaussjacobi(Np,0.0,0.0)
    dg.r_basis,dg.w_basis=FastGaussQuadrature.gausslobatto(dg.Np)

    dg.VX = range(domain_x_limits[1],domain_x_limits[2], N_elem_per_dim+1) |> collect
    dg.delta_x = dg.VX[2]-dg.VX[1]
    # constant jacobian on all elements as they are evenly spaced
    jacobian = dg.delta_x/2 #reference element is 2 units long
    dg.J = LinearAlgebra.diagm(ones(size(dg.r_volume))*jacobian)

    dg.x = ones(dg.N_elem_per_dim*dg.Np)
    for i_elem = 1:N_elem_per_dim
        x_local = dg.VX[i_elem] .+ 0.5* (dg.r_volume .+1) * dg.delta_x
        dg.x[dg.EIDLIDtoGID[i_elem,1:dg.Np]] .= x_local
    end

    # Define Vandermonde matrices
    dg.chi_v = vandermonde1D(dg.r_volume,dg.r_basis)
    dg.d_chi_v_d_xi = gradvandermonde1D(dg.r_volume,dg.r_basis)
    #reference coordinates of L and R faces
    r_f_L::Float64 = -1
    r_f_R::Float64 = 1
    dg.chi_f = assembleFaceVandermonde1D(r_f_L,r_f_R,dg.r_basis)

    dg.W = LinearAlgebra.diagm(dg.w) # diagonal matrix holding quadrature weights
    dg.W_f = 1.0
    
    # Mass and stiffness matrices as defined Eq. 9.5 Cicchino 2022
    # All defined on a single element.
    dg.M = dg.chi_v' * dg.W * dg.J * dg.chi_v
    dg.M_inv = inv(dg.M)
    dg.S_xi = dg.chi_v' * dg.W * dg.d_chi_v_d_xi
    dg.S_noncons = dg.W * dg.d_chi_v_d_xi
    dg.M_nojac = dg.chi_v' * dg.W * dg.chi_v
    dg.Pi = inv(dg.M_nojac)*dg.chi_v'*dg.W



    return dg

end

#init_DG(4,1,5,  [0.0, 1.0] )
