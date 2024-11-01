#==============================================================================
# Polynomial functions to assemble Vandermonde matrix and grad of Vandermonde matrix
==============================================================================#


include("set_up_dg.jl")

function lagrangep(r_volume,r_basis,j)
    #return coeffs of j-th lagrange polynomial constructed with nodes r and evaluate at nodes r
    #
    Np_cubature = length(r_volume)
    Np_basis = length(r_basis)
    lagrange_p::Vector{Float64} = ones(Np_cubature) # length N+1
    for i in 1:Np_cubature
        for m in 1:Np_basis
           if m != j
               lagrange_p[i] *= (r_volume[i] - r_basis[m])/(r_basis[j]-r_basis[m])
           end
        end
    end
    return lagrange_p
end

function vandermonde1D(r_volume::AbstractVector, r_basis::AbstractVector)
    V = zeros(Float64,(length(r_volume),length(r_basis)))
    for j in 1:length(r_basis)
        V[:,j] .= lagrangep(r_volume, r_basis, j)
    end
    return V
end

function tensor_product_vandermonde2D(r_basis::AbstractVector, r_volume_x::AbstractVector, r_volume_y::AbstractVector, V1Dx::AbstractMatrix, V1Dy::AbstractMatrix)
    #Helper function to find the tensor product 2D vandermonde matrix from 1D vandermonde matrices in each direction. 
    #This function assumes that the same basis function is used in  each coordinate direction, but not necessarily the same volume integration
    #allowing for face VdM to be assembled.

    V2D = zeros(Float64, (length(r_volume_x)*length(r_volume_y), length(r_basis)^2))
    i_LID_V = 1
    for iy_V = 1:length(r_volume_y)
        for ix_V = 1:length(r_volume_x)
            i_LID_B = 1
            for iy_B = 1:length(r_basis)
                for ix_B = 1:length(r_basis)
                    V2D[i_LID_V, i_LID_B] = V1Dx[ix_V, ix_B] * V1Dy[iy_V, iy_B]
                    i_LID_B+=1
                end
            end
            i_LID_V += 1
        end
    end

    return V2D

end

function vandermonde2D(r_volume::AbstractVector, r_basis::AbstractVector, dg::DG)
    V1D = vandermonde1D(r_volume::AbstractVector, r_basis::AbstractVector)
    return tensor_product_vandermonde2D(r_basis, r_volume, r_volume, V1D, V1D)
end

function gradlagrangep(r_volume,r_basis,j)
    #return coeffs of j-th lagrange polynomial constructed with nodes r and evaluate at nodes r
    #
    Np_cubature = length(r_volume)
    Np_basis = length(r_basis)
    dlagrange_p::Vector{Float64} = zeros(Np_cubature) # length N+1
    for ind_x in 1:Np_cubature
        x = r_volume[ind_x]
        l_prime_at_x = 0
        for i in 1:Np_basis
            if i != j
                prod = 1
                for m in 1:Np_basis
                    if (m!=i) & (m != j)
                        prod *= (x - r_basis[m])/(r_basis[j]-r_basis[m])
                    end
                end
                l_prime_at_x += 1/(r_basis[j]-r_basis[i])*prod
            end
        end
        dlagrange_p[ind_x] = l_prime_at_x
    end

    return dlagrange_p
end

function gradvandermonde1D(r_volume::AbstractVector, r_basis::AbstractVector)
    DVr::Matrix{Float64} = zeros(Float64,(length(r_volume),length(r_basis)))
    Np_basis = length(r_basis)
    for j in 1:Np_basis
        DVr[:,j] .= gradlagrangep(r_volume, r_basis, j)
    end
    return DVr
end


function gradvandermonde2D(direction::Int, r_volume::AbstractVector, r_basis::AbstractVector, dg::DG)

    DVr1D = gradvandermonde1D(r_volume::AbstractVector, r_basis::AbstractVector)
    V1D = vandermonde1D(r_volume::AbstractVector, r_basis::AbstractVector)

    if direction == 1
        return tensor_product_vandermonde2D(r_basis, r_volume, r_volume, DVr1D, V1D)
    else
        return tensor_product_vandermonde2D(r_basis, r_volume, r_volume, V1D, DVr1D)
    end
end

function assembleFaceVandermonde1D(r_f_L::Float64, r_f_R::Float64, r_basis::AbstractVector)
    V_f = zeros(Float64, (1,length(r_basis), 2)) #third dimension is face ID
    V_f[:,:,1] .= vandermonde1D([r_f_L], r_basis)
    V_f[:,:,2] .= vandermonde1D([r_f_R], r_basis)
    return V_f
end

function assembleFaceVandermonde2D(r_basis::AbstractVector, r_volume::AbstractVector, dg::DG)
    
    V_f = zeros(Float64, (length(r_volume),length(r_basis)^dg.dim, 4)) #third dimension is face ID
    # stores vandermonde matrices evaluated at -1 and 1
    V_f_1D = assembleFaceVandermonde1D(-1.0, 1.0, r_basis)
    # stores a 1D vandermonde matrix at volume integration points
    V_v_1D = vandermonde1D(r_volume::AbstractVector, r_basis::AbstractVector)
    
    #This implementation assumes that face points are a subset of volume points.
    
    for iface = 1:dg.N_faces
        if iface == 1
            r_face_y = r_volume
            r_face_x = [-1.0]
            V_x = V_f_1D[:,:,1] #-1 face
            V_y = V_v_1D
        elseif iface == 2
            r_face_y = r_volume
            r_face_x = [1.0]
            V_x = V_f_1D[:,:,2] #1 face
            V_y = V_v_1D
        elseif iface == 3
            r_face_x = r_volume
            r_face_y = [-1.0]
            V_y = V_f_1D[:,:,1] #-1 face
            V_x = V_v_1D
        elseif iface == 4
            r_face_x = r_volume
            r_face_y = [1.0]
            V_y = V_f_1D[:,:,2] #1 face
            V_x = V_v_1D
        end
        V_f[:,:,iface] .= tensor_product_vandermonde2D(r_basis, r_face_x, r_face_y, V_x, V_y)
    end
    return V_f
end
