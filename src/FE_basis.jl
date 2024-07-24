#==============================================================================
# Polynomial functions to assemble Vandermonde matrix and grad of Vandermonde matrix
==============================================================================#


include("set_up_dg.jl")

function lagrangep(r_cubature,r_basis,j)
    #return coeffs of j-th lagrange polynomial constructed with nodes r and evaluate at nodes r
    #
    Np_cubature = length(r_cubature)
    Np_basis = length(r_basis)
    lagrange_p::Vector{Float64} = ones(Np_cubature) # length N+1
    for i in 1:Np_cubature
        for m in 1:Np_basis
           if m != j
               lagrange_p[i] *= (r_cubature[i] - r_basis[m])/(r_basis[j]-r_basis[m])
           end
        end
    end
    return lagrange_p
end

function vandermonde1D(r_cubature::AbstractVector, r_basis::AbstractVector)
    V = zeros(Float64,(length(r_cubature),length(r_basis)))
    for j in 1:length(r_basis)
        V[:,j] .= lagrangep(r_cubature, r_basis, j)
    end
    return V
end

function vandermonde2D(r_cubature::AbstractVector, r_basis::AbstractVector, dg::DG)

    V1D = vandermonde1D(r_cubature::AbstractVector, r_basis::AbstractVector)

    dim = 2
    V2D = zeros(Float64, (length(r_cubature)^dim, length(r_basis)^dim))
    for iy_C = 1:length(r_cubature)
        for ix_C = 1:length(r_cubature)
            i_LID_C = dg.LXIDLYIDtoLID[ix_C,iy_C]

            counter_basis = 1
            for iy_B = 1:length(r_basis)
                for ix_B = 1:length(r_basis)
                    i_LID_B = counter_basis
                    V2D[i_LID_C, i_LID_B] = V1D[ix_C,ix_B]*V1D[iy_C,iy_B]
                    counter_basis = counter_basis + 1
                end
            end
        end
        
    end
    return V2D
end

function gradlagrangep(r_cubature,r_basis,j)
    #return coeffs of j-th lagrange polynomial constructed with nodes r and evaluate at nodes r
    #
    Np_cubature = length(r_cubature)
    Np_basis = length(r_basis)
    dlagrange_p::Vector{Float64} = zeros(Np_cubature) # length N+1
    for ind_x in 1:Np_cubature
        x = r_cubature[ind_x]
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

function gradvandermonde1D(r_cubature::AbstractVector, r_basis::AbstractVector)
    DVr::Matrix{Float64} = zeros(Float64,(length(r_cubature),length(r_basis)))
    Np_basis = length(r_basis)
    for j in 1:Np_basis
        DVr[:,j] .= gradlagrangep(r_cubature, r_basis, j)
    end
    return DVr
end
function gradvandermonde2D(direction::Int, r_cubature::AbstractVector, r_basis::AbstractVector, dg::DG)

    DVr1D = gradvandermonde1D(r_cubature::AbstractVector, r_basis::AbstractVector)
    V1D = gradvandermonde1D(r_cubature::AbstractVector, r_basis::AbstractVector)

    dim = 2
    DVr2D = zeros(Float64, (length(r_cubature)^dim, length(r_basis)^dim))
    for iy_C = 1:length(r_cubature)
        for ix_C = 1:length(r_cubature)
            i_LID_C = dg.LXIDLYIDtoLID[ix_C,iy_C]

            counter_basis = 1
            for iy_B = 1:length(r_basis)
                for ix_B = 1:length(r_basis)
                    i_LID_B = counter_basis
                    if direction == 1 #derivative wrt xi, first direction
                        DVr2D[i_LID_C, i_LID_B] = DVr1D[ix_C,ix_B]*V1D[iy_C,iy_B]
                    elseif direction == 2 #derivative wrt eta, first direction
                        DVr2D[i_LID_C, i_LID_B] = V1D[ix_C,ix_B]*DVr1D[iy_C,iy_B]
                    end
                    counter_basis = counter_basis + 1
                end
            end
        end
        
    end
    return DVr2D
end

function assembleFaceVandermonde1D(r_f_L::Float64, r_f_R::Float64, r_basis::AbstractVector)
    V_f = zeros(Float64, (1,length(r_basis), 2)) #third dimension is face ID
    V_f[:,:,1] .= vandermonde1D([r_f_L], r_basis)
    V_f[:,:,2] .= vandermonde1D([r_f_R], r_basis)
    return V_f
end

function assembleFaceVandermonde2D(chi_v,r_basis::AbstractVector, dg)
    V_f = zeros(Float64, (length(dg.r_volume),length(r_basis)^dg.dim, 4)) #third dimension is face ID
    
    #This implementation assumes that face points are a subset of volume points.
    for iface = 1:dg.Nfaces
        #get LID of points on LFID of iface
        LID_iface = dg.LFIDtoLID[iface,:]
        for ifacept = 1:dg.Nfp
            V_f[ifacept,:,iface] = chi_v[LID_iface[ifacept], :]
        end
    end
    return V_f
end
