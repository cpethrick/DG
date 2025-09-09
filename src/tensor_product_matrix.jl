# Type for a matrix which is defined as a tensor product of 1D bases

include("operators.jl")
include("FE_basis.jl")
import FastGaussQuadrature
import Random
import LinearAlgebra
import PyPlot

mutable struct TensorProductMatrix2D
    # Assume matrix is 2D tensor product.

    basis1::AbstractMatrix # basis in first dimension
    basis2::AbstractMatrix # basis in second dimension
    M::AbstractMatrix # store M = basis1 ⊗ basis2


    # incomplete initialization: take only bases.
    TensorProductMatrix2D(basis1::AbstractMatrix,
                          basis2::AbstractMatrix) = new(basis1::AbstractMatrix,basis2::AbstractMatrix)
end


function init_TensorProductMatrix2D(basis1::AbstractMatrix,
                                    basis2::AbstractMatrix)
    A = TensorProductMatrix2D(basis1,basis2)
    A.M = tensor_product_2D(basis1,basis2)

    return A
end


##### Define operations 
# These return a type AbstractMatrix for now, rather than a new TensorProductMatrix2D.
function Base.:+(a::TensorProductMatrix2D, b::TensorProductMatrix2D)
    return (a.M + b.M)
end 


#Sum-factorized matrix-vector multiplication
function matrix_vector_mult(A::TensorProductMatrix2D,x::AbstractVector,N::Int64)
    # Calculates Ax using sum factorization for A ∈ R^NxN and x ∈ N^2
    #
    # This is done in one line to minimize storage accesses.
    # The procedure is:
    #   1. Rearrange the vector x from an N^2 vector to a NxN matrix, ordered like
    #       [1 3
    #       2 4]
    #   2. Apply basis 1: temp = A.basis1 * x_matrix
    #   3. Apply basis 2: b_matrix = A.basis2 * temp'
    #   4. Reshape b_matrix into an N^2 vector.

    return reshape((A.basis2*(A.basis1*reshape(x, (N,N)))')',N*N)

end


function test_tensor_product(N)
    # Integration points
    
    (r,w) = FastGaussQuadrature.gausslobatto(N) 

    vdm_1D = vandermonde1D(r,r)
    M_1D = vdm_1D'*LinearAlgebra.diagm(w) * vdm_1D

    # Initialize a mass matrix
    mass_matrix = init_TensorProductMatrix2D(M_1D,M_1D)

    #display(mass_matrix.basis1)
    #display(mass_matrix.basis2)
    #display(mass_matrix.M)

    # Initialize a solution-like vector

    soln = Random.rand!(zeros(N*N))
    #display(soln)

    #Naive implementation
    naive_time = @elapsed begin
        x_naive = mass_matrix.M * soln
    end
    #display("naive M*u")
    #display(x_naive)


    # Tensor product implementation
    tensor_prod_time = @elapsed begin
        #rearrange soln vector into a matrix
        #soln_mat = reshape(soln, (N,N)) # reshapes like [1 3; 2 4]
        #x_mat = (mass_matrix.basis2*(mass_matrix.basis1*soln_mat)')
        #x_tensor_prod = reshape((mass_matrix.basis2*(mass_matrix.basis1*soln_mat)')',N*N)
        #one line
        #x_tensor_prod = reshape((mass_matrix.basis2*(mass_matrix.basis1*reshape(soln, (N,N)))')',N*N)
        x_tensor_prod = matrix_vector_mult(mass_matrix,soln,N)

    end
    #display("tensor prod M*u")
    #display(x_tensor_prod)

    if any( val >1E-14 for val in (abs.(x_naive-x_tensor_prod)))
        display("Results do not match!!")
    end

    return (naive_time, tensor_prod_time)
end


test_tensor_product(4)

end_number = 30
result = zeros(end_number,3)
for N in range(2,end_number+1)
    (timing_naive,timing_tensor_prod) = (0,0)
    for i in 1:10
        (timing_n,timing_p) = test_tensor_product(N)
        timing_naive += timing_n
        timing_tensor_prod += timing_p
    end
   result[N-1,1] = N
   result[N-1,2] = timing_naive
   result[N-1,3] = timing_tensor_prod
end
display("N, naive, tensor-prod")
display(result)

PyPlot.clf()
PyPlot.plot(result[:,1], result[:,2:3])
PyPlot.legend(["Naive implementation", "Tensor product"])
