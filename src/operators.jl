
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

function hadamard_product(A::AbstractMatrix, B::AbstractMatrix)
    #returns C = A ⊙ B, element-wise multiplication of matrices A and B
    #no sum factorization.
    
    N_rows = size(A)[1]
    N_cols = size(A)[2]

    if (size(A) != size(B))
        display("Error! Hadamard product assumes that inputs A and B are the same size.")
        return 0
    end
    
    C = zeros(N_rows,N_cols)
    for irow = 1:N_rows
        for icol = 1:N_cols
            C[irow,icol] = A[irow,icol] * B[irow,icol]
        end
    end

    return C
    
end
