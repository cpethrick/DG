#==============================================================================
# Polynomial functions to assemble Vandermonde matrix and grad of Vandermonde matrix
==============================================================================#


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

function assembleFaceVandermonde1D(r_f_L::Float64, r_f_R::Float64, r_basis::AbstractVector)
    V_f = zeros(Float64, (1,length(r_basis), 2)) #third dimension is face ID
    V_f[:,:,1] .= vandermonde1D([r_f_L], r_basis)
    V_f[:,:,2] .= vandermonde1D([r_f_R], r_basis)
    return V_f
end
